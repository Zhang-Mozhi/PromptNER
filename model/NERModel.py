from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
from torch_scatter import scatter_max, scatter_mean
import torch
import torch.nn.functional as F
from .cnn import MaskCNN
from .multi_head_biaffine import MultiHeadBiaffine
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import math

from matplotlib import pyplot as plt
import pandas as pd
from sklearn import manifold


BertHiddenSize = 768


class PromptNER(nn.Module):
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=0, cnn_depth=0, N=5, K=1, mode='inter', isCrossNER=False, kNN_ratio=2.0):
        super(PromptNER, self).__init__()
        self.model_name = model_name
        self.bert_config = BertConfig.from_pretrained(model_name)
        self.K = K
        self.N = N
        self.kNN_ratio = kNN_ratio
        self.pretrain_model = AutoModel.from_pretrained(model_name, self.bert_config)
        if self.K == 5 and (mode == '4' or mode == 'inter' or mode == 'intra'):
            self.pretrain_model.config.gradient_checkpointing = True
      
        self.proto_model = AutoModel.from_pretrained(model_name, self.bert_config)
        if self.K == 5 and (mode == '4' or mode == 'inter' or mode == 'intra'):
            self.proto_model.config.gradient_checkpointing = True
        self.isCrossNER = isCrossNER

        self.train_threshold = 0.5
        self.predict_threshold = 0.2
        self.max_span_len = 14
        self.hidden_size = self.pretrain_model.config.hidden_size
        self.cnn_dim = cnn_dim
        self.biaffine_size = biaffine_size

        self.mode = mode # for Few-NERD
        self.RoPE = True

        if self.K == 1:
            self.mention_num = 1
            if self.mode == 'inter':
                self.predict_num = 5
            else:
                self.predict_num = 3
        else:
            self.mention_num = 5
            if self.mode == 'inter':
                self.predict_num = 15
            else:
                self.predict_num = 15

        if self.mode == 'inter':
            self.beta = 0.5
        else:
            self.beta = 0.3
        #self.beta = 1.0 # w/o rerank ablation

        if size_embed_dim != 0:
            n_pos = 128
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        biaffine_input_size = self.hidden_size

        self.head_mlp = nn.Sequential(
            #nn.Dropout(0.05),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            #nn.Dropout(0.05),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(0.1)
        if n_head > 0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth > 0:
            # self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=cnn_depth)
            self.cnn = None

        # Span_detector
        self.down_fc = nn.Linear(cnn_dim, 1)
        
        self.down_layer = nn.Sequential(
            nn.Linear(cnn_dim, 1),
        )
        self.dense = nn.Linear(self.hidden_size, self.biaffine_size)
        self.logit_drop = logit_drop

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward_v0(self, state, indices): 
        """
        Biaffine + N x CNN module
        state: bsz x seq_len x hsz
        """
        lengths, _ = indices.max(dim=-1)
        bsz, seq_len, hsz = state.size()
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)],
                                     dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
     
        RoPE_logits = 0
        if self.RoPE:
            outputs = self.dense(state)
            outputs = torch.split(outputs, self.biaffine_size, dim=-1)
            # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
            outputs = torch.stack(outputs, dim=-2)
            # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
            qw, kw = outputs[...,:self.biaffine_size // 2], outputs[...,self.biaffine_size // 2:] 
            pos_emb = self.sinusoidal_position_embedding(bsz, seq_len, self.biaffine_size // 2, state.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            RoPE_logits= torch.einsum('bmhd,bnhd->bhmn', qw, kw).permute(0, 2, 3, 1).squeeze(-1)

        scores = scores2 + scores1  # bsz x dim x L x L
   
        if hasattr(self, 'cnn'):
            mask = seq_len_to_mask(lengths)  # bsz x length x length
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(u_scores, p=self.logit_drop, training=self.training)
            # bsz, num_label, max_len, max_len = u_scores.size()
            u_scores = self.cnn(u_scores, pad_mask)
            scores = u_scores + scores
        scores = scores.permute(0, 2, 3, 1)

        return RoPE_logits / (self.biaffine_size ** 0.5) 
    
    def forward(self, state, indices): 
        """
        state: bsz x seq_len x hsz
        """
        lengths, _ = indices.max(dim=-1)
        bsz, seq_len, hsz = state.size()
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)
     
        RoPE_logits = 0
        if self.RoPE:
            outputs = self.dense(state)
            outputs = torch.split(outputs, self.biaffine_size, dim=-1)
            # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
            outputs = torch.stack(outputs, dim=-2)
            # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
            qw, kw = outputs[...,:self.biaffine_size // 2], outputs[...,self.biaffine_size // 2:] 
            pos_emb = self.sinusoidal_position_embedding(bsz, seq_len, self.biaffine_size // 2, state.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            RoPE_logits= torch.einsum('bmhd,bnhd->bhmn', qw, kw).permute(0, 2, 3, 1).squeeze(-1)
      
        scores = scores1  # bsz x biaffine_dim x L x L

        scores = scores.permute(0, 2, 3, 1) # bsz x L x L x biaffine_dim 
 
        return (self.down_layer(scores).squeeze(-1) + RoPE_logits) / (self.biaffine_size ** 0.5)   # bsz x L x L
     
    def get_span_rep(self, emb, indices):
        """
        Args:
            emb: seq_len x hidden_size
            indices: 2 x span_num_of_each_sent
        Returns:
            emb: span_num_of_each_sent x hidden_size
        """
        device = emb.device

        span_reps = []
        span_num = indices[0].size(0)
    
        for index in range(span_num):
            span_rep = emb[indices[0][index]:indices[1][index]+1, :].mean(dim=0)
            span_reps.append(span_rep)

        span_reps = torch.stack(span_reps, dim=0).to(device)

        return span_reps

    def train_step_batch_NN(self, support, query):
        device = support['prefix_prompt_word'].device

        support_num = query_num = 0
        support_gold_num = query_gold_num = 0

        support_max_seq_len, query_max_seq_len = max(support['seq_len']), max(query['seq_len'])

        loss = 0
        support_CL_loss = 0
        query_CL_loss = 0
        support_gold_loss = 0
        query_gold_loss = 0
        Type_loss = 0
        Span_loss = 0
        bs = 0

        for support_cur_num, query_cur_num in zip(support['sentence_num'], query['sentence_num']):
            support_attention_mask = seq_len_to_mask(torch.tensor(support['seq_len'][support_num:
                                                                                     support_num + support_cur_num]),
                                                     max_len=support_max_seq_len).int().to(device)
            
            support_out = self.pretrain_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            """
            support_out = self.pretrain_model(support['word'][support_num: support_num + support_cur_num],
                                              attention_mask=support_attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            """

            support_out_1 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768  

            query_attention_mask = seq_len_to_mask(torch.tensor(query['seq_len'][query_num:query_num + query_cur_num]),
                                                   max_len=query_max_seq_len).int().to(device)
            
            query_out = self.pretrain_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            query_out = self.pretrain_model(query['word'][query_num: query_num + query_cur_num],
                                            attention_mask=query_attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            query_out_1 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                           output_hidden_states=True,
                                           return_dict=True)  # num_sent x number_of_tokens x 768
        
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])
            query_cur_seq_len = max(query['seq_len'][query_num: query_num + query_cur_num])
            query_cur_prefix_seq_len = max(query['prefix_seq_len'][query_num: query_num + query_cur_num])

            #support_emb = support_out['last_hidden_state'][:, 0:support_cur_seq_len, :]
            support_emb = support_out['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
            support_emb_1 = support_out_1['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]

            #query_emb = query_out['last_hidden_state'][:, 0:query_cur_seq_len, :]
            query_emb = query_out['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
            query_emb_1 = query_out_1['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]


            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            query_indices = query['indices'][query_num:query_num + query_cur_num][:, 0:query_cur_seq_len]

            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]
            query_prefix_indices = query['prefix_indices'][query_num:query_num + query_cur_num][:, 0:query_cur_prefix_seq_len]

            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            query_gold_span_num = query['gold_span_num'][query_num:query_num + query_cur_num]
            query_cur_gold_num = sum(query_gold_span_num)

            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)
            query_gold_span_tag = query['gold_span_tag'][query_gold_num: query_gold_num + query_cur_gold_num].to(device)
            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            query_gold_spans = query['gold_spans'][query_num:query_num + query_cur_num]

            self.N = support_gold_span_tag.max()
            #support_emb = scatter_mean(support_emb, index=support_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
            support_label_emb = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, 1:self.N + 2]
            support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        
            #query_emb = scatter_mean(query_emb, index=query_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            query_emb = scatter_mean(query_emb, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
            query_label_emb = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, 1:self.N + 2]
            query_emb_1 = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

            support_bert_emb = support_emb_1
            query_bert_emb = query_emb_1
            
            if self.mode == 'inter' or self.mode == 'intra':
                if self.K == 5:
                    support_bert_emb_CL = support_emb_1
                    query_bert_emb_CL = query_emb_1
                else: 
                    support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                                    output_hidden_states=True,
                                                    return_dict=True)  # num_sent x number_of_tokens x 768              

                    support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                    support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]   # bsz x seq_len x hidden_size
                    support_bert_emb_CL = support_emb_2

                    query_out_2 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                                    output_hidden_states=True,
                                                    return_dict=True)  # num_sent x number_of_tokens x 768    
                    query_emb_2 = query_out_2['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
                    query_emb_2 = scatter_mean(query_emb_2, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

                    query_bert_emb_CL = query_emb_2
            else:
                if self.K == 5:
                     support_bert_emb_CL = support_emb_1
                else:
                    support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                                    output_hidden_states=True,
                                                    return_dict=True)  # num_sent x number_of_tokens x 768              

                    support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                    support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]   # bsz x seq_len x hidden_size
                    support_bert_emb_CL = support_emb_2
                query_out_2 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                                    output_hidden_states=True,
                                                    return_dict=True)  # num_sent x number_of_tokens x 768    
                query_emb_2 = query_out_2['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
                query_emb_2 = scatter_mean(query_emb_2, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

                query_bert_emb_CL = query_emb_2  
            # Biaffine Module
            support_span_scores = self(support_emb, support_indices)  # bsz x seq_len x seq_len x dim
            query_span_scores = self(query_emb, query_indices)  # bsz x seq_len x seq_len x dim

            support_span_loss = self.Span_Loss_V2(support_span_scores, support_gold_spans, support_indices)
            query_span_loss = self.Span_Loss_V2(query_span_scores, query_gold_spans, query_indices)

            support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            support_gold_rep_CL = self.get_gold_span_rep(support_bert_emb_CL, support_gold_spans, support_gold_span_tag)  # gold_num x 768


            cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep_CL, support_gold_span_tag)
            #cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep, support_gold_span_tag)
            if (self.N == 10 and self.K == 5) or (self.K == 5 and self.isCrossNER):
                cur_query_gold_loss = 0
            else:
                query_gold_rep = self.get_gold_span_rep(query_bert_emb, query_gold_spans, query_gold_span_tag)  # gold_num x 768
                query_gold_rep_CL = self.get_gold_span_rep(query_bert_emb_CL, query_gold_spans, query_gold_span_tag)  # gold_num x 768
                if self.mode == 'inter' or self.mode == 'intra':
                    cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep_CL, query_gold_span_tag)
                else:
                    cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep_CL, query_gold_span_tag, notQueryInSnips=False)
            #cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep, query_gold_span_tag)

            """
            cur_query_type_loss = self.Type_Loss(query_span_scores.detach(), query_bert_emb, 
                                                    support_gold_span_tag, query_gold_spans, query_gold_span_tag,
                                                    query_indices, query_label_emb)
        
            cur_support_type_loss = self.Type_Loss(support_span_scores.detach(), support_bert_emb, 
                                                    support_gold_span_tag, support_gold_spans, support_gold_span_tag,
                                                    support_indices, support_label_emb)
            """

            cur_query_type_loss = self.Type_Loss(query_bert_emb, support_gold_span_tag, query_gold_spans, query_gold_span_tag,
                                                query_indices, query_label_emb)
        
            cur_support_type_loss = self.Type_Loss(support_bert_emb, support_gold_span_tag, support_gold_spans, support_gold_span_tag,
                                                    support_indices, support_label_emb)
            
            bs += 1
            if (self.N == 10 and self.K == 5) or (self.K == 5 and self.isCrossNER):
                loss = loss + (cur_support_type_loss + cur_query_type_loss) / 2 + (support_span_loss + query_span_loss) / 2# + cur_support_gold_loss 
            else:
                loss = loss + (cur_support_type_loss + cur_query_type_loss) / 2 + (support_span_loss + query_span_loss) / 2 #+ (cur_support_gold_loss + cur_query_gold_loss) / 2
            #loss = loss +  cur_query_type_loss + (support_span_loss + query_span_loss) / 2  + (cur_support_gold_loss + cur_query_gold_loss) / 2
            Type_loss += (cur_support_type_loss + cur_query_type_loss) / 2
            #Type_loss = cur_query_type_loss
            Span_loss += (support_span_loss + query_span_loss) / 2

            support_gold_loss += support_span_loss  # cur_support_gold_loss
            query_gold_loss += query_span_loss
            support_CL_loss += cur_support_gold_loss
            query_CL_loss += cur_query_gold_loss

            support_gold_num += support_cur_gold_num
            query_gold_num += query_cur_gold_num
            support_num += support_cur_num
            query_num += query_cur_num

        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        support_CL_loss /= bs
        query_CL_loss /= bs
        support_gold_loss /= bs
        query_gold_loss /= bs
    

        return {'loss': loss, "Type_loss": Type_loss, "Span_loss":Span_loss, "support_CL_loss": support_CL_loss, "query_CL_loss": query_CL_loss,
                "CL_loss": support_CL_loss + query_CL_loss, "support_gold_loss": support_gold_loss,
                "query_gold_loss": query_gold_loss, "Second_stage_loss": Type_loss + support_CL_loss + query_CL_loss}
    
    def train_step_stage_one(self, support, query):
        device = support['prefix_prompt_word'].device

        support_num = query_num = 0
        support_gold_num = query_gold_num = 0

        support_max_seq_len, query_max_seq_len = max(support['seq_len']), max(query['seq_len'])

        loss = 0
        support_CL_loss = 0
        query_CL_loss = 0
        support_gold_loss = 0
        query_gold_loss = 0
        Type_loss = 0
        Span_loss = 0
        bs = 0

        for support_cur_num, query_cur_num in zip(support['sentence_num'], query['sentence_num']):
            support_attention_mask = seq_len_to_mask(torch.tensor(support['seq_len'][support_num:
                                                                                     support_num + support_cur_num]),
                                                     max_len=support_max_seq_len).int().to(device)
           
            support_out = self.pretrain_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            """
            support_out = self.pretrain_model(support['word'][support_num: support_num + support_cur_num],
                                              attention_mask=support_attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            """
           
            query_attention_mask = seq_len_to_mask(torch.tensor(query['seq_len'][query_num:query_num + query_cur_num]),
                                                   max_len=query_max_seq_len).int().to(device)
           
            query_out = self.pretrain_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            query_out = self.pretrain_model(query['word'][query_num: query_num + query_cur_num],
                                            attention_mask=query_attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])
            query_cur_seq_len = max(query['seq_len'][query_num: query_num + query_cur_num])
            query_cur_prefix_seq_len = max(query['prefix_seq_len'][query_num: query_num + query_cur_num])

            #support_emb = support_out['last_hidden_state'][:, 0:support_cur_seq_len, :]
            support_emb = support_out['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]

            #query_emb = query_out['last_hidden_state'][:, 0:query_cur_seq_len, :]
            query_emb = query_out['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]

            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            query_indices = query['indices'][query_num:query_num + query_cur_num][:, 0:query_cur_seq_len]
     
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]
            query_prefix_indices = query['prefix_indices'][query_num:query_num + query_cur_num][:, 0:query_cur_prefix_seq_len]

            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            query_gold_spans = query['gold_spans'][query_num:query_num + query_cur_num]
           
            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            query_gold_span_num = query['gold_span_num'][query_num:query_num + query_cur_num]
            query_cur_gold_num = sum(query_gold_span_num)
            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)
            self.N = support_gold_span_tag.max()

            #support_emb = scatter_mean(support_emb, index=support_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2::]  # bsz x seq_len x hidden_size
            #query_emb = scatter_mean(query_emb, index=query_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            query_emb = scatter_mean(query_emb, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

            support_span_scores = self(support_emb, support_indices)  # bsz x seq_len x seq_len x dim

            query_span_scores = self(query_emb, query_indices)  # bsz x seq_len x seq_len x dim


            support_span_loss = self.Span_Loss_V2(support_span_scores, support_gold_spans, support_indices)
            query_span_loss = self.Span_Loss_V2(query_span_scores, query_gold_spans, query_indices)

            bs += 1

            loss = loss + (support_span_loss + query_span_loss) 

            Span_loss += (support_span_loss + query_span_loss)

            support_gold_loss += support_span_loss  # cur_support_gold_loss
            query_gold_loss += query_span_loss

            support_gold_num += support_cur_gold_num
            query_gold_num += query_cur_gold_num
            support_num += support_cur_num
            query_num += query_cur_num

        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        support_CL_loss /= bs
        query_CL_loss /= bs
        support_gold_loss /= bs
        query_gold_loss /= bs
    
        return {'loss': loss, "Type_loss": Type_loss, "Span_loss":Span_loss, "support_CL_loss": support_CL_loss, "query_CL_loss": query_CL_loss,
                "CL_loss": support_CL_loss + query_CL_loss, "support_gold_loss": support_gold_loss,
                "query_gold_loss": query_gold_loss, "Second_stage_loss": Type_loss + support_CL_loss + query_CL_loss}
    
    def train_step_stage_two_V0(self, support, query):
        device = support['prefix_prompt_word'].device

        support_num = query_num = 0
        support_gold_num = query_gold_num = 0

        #support_max_seq_len, query_max_seq_len = max(support['seq_len']), max(query['seq_len'])

        loss = 0
        support_CL_loss = 0
        query_CL_loss = 0
        support_gold_loss = 0
        query_gold_loss = 0
        Type_loss = 0
        bs = 0

        for support_cur_num, query_cur_num in zip(support['sentence_num'], query['sentence_num']):
            
            support_out = self.pretrain_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768    
            """
            support_attention_mask = seq_len_to_mask(torch.tensor(support['seq_len'][support_num: support_num + support_cur_num]),
                                                     max_len=support_max_seq_len).int().to(device)

            support_out = self.pretrain_model(support['word'][support_num: support_num + support_cur_num],
                                              attention_mask=support_attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            """
            
            support_out_1 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768  


            
            query_out = self.pretrain_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            query_attention_mask = seq_len_to_mask(torch.tensor(query['seq_len'][query_num:query_num + query_cur_num]),
                                                   max_len=query_max_seq_len).int().to(device)
            query_out = self.pretrain_model(query['word'][query_num: query_num + query_cur_num],
                                            attention_mask=query_attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            
            query_out_1 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                           output_hidden_states=True,
                                           return_dict=True)  # num_sent x number_of_tokens x 768
        
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])
            query_cur_seq_len = max(query['seq_len'][query_num: query_num + query_cur_num])
            query_cur_prefix_seq_len = max(query['prefix_seq_len'][query_num: query_num + query_cur_num])

            #support_emb = support_out['last_hidden_state'][:, 0:support_cur_seq_len, :]
            support_emb = support_out['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
            support_emb_1 = support_out_1['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]

            #query_emb = query_out['last_hidden_state'][:, 0:query_cur_seq_len, :]
            query_emb = query_out['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
            query_emb_1 = query_out_1['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]


            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            query_indices = query['indices'][query_num:query_num + query_cur_num][:, 0:query_cur_seq_len]
     
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]
            query_prefix_indices = query['prefix_indices'][query_num:query_num + query_cur_num][:, 0:query_cur_prefix_seq_len]

            #support_emb = scatter_mean(support_emb, index=support_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2::]  # bsz x seq_len x hidden_size
            support_label_emb = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, 1:self.N + 2]
            support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        
            #query_emb = scatter_mean(query_emb, index=query_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            query_emb = scatter_mean(query_emb, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
            query_label_emb = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, 1:self.N + 2]
            query_emb_1 = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

            support_bert_emb = support_emb_1
            query_bert_emb = query_emb_1
            
            if self.K == 5:
                support_bert_emb_CL = support_emb_1
                query_bert_emb_CL = query_emb_1
            else: 
                support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768              

                support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]   # bsz x seq_len x hidden_size
                support_bert_emb_CL = support_emb_2

                query_out_2 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768    
                query_emb_2 = query_out_2['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
                query_emb_2 = scatter_mean(query_emb_2, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

                query_bert_emb_CL = query_emb_2

            # Biaffine Module
            support_span_scores = self(support_emb, support_indices)  # bsz x seq_len x seq_len x dim
            query_span_scores = self(query_emb, query_indices)  # bsz x seq_len x seq_len x dim

            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            query_gold_spans = query['gold_spans'][query_num:query_num + query_cur_num]

            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            query_gold_span_num = query['gold_span_num'][query_num:query_num + query_cur_num]
            query_cur_gold_num = sum(query_gold_span_num)

            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)
            query_gold_span_tag = query['gold_span_tag'][query_gold_num: query_gold_num + query_cur_gold_num].to(device)

            support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            support_gold_rep_CL = self.get_gold_span_rep(support_bert_emb_CL, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            
            query_gold_rep = self.get_gold_span_rep(query_bert_emb, query_gold_spans, query_gold_span_tag)  # gold_num x 768
            query_gold_rep_CL = self.get_gold_span_rep(query_bert_emb_CL, query_gold_spans, query_gold_span_tag)  # gold_num x 768

            cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep_CL, support_gold_span_tag)

            cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep_CL, query_gold_span_tag)

            cur_query_type_loss = self.Type_Loss_V0(query_span_scores.detach(), query_bert_emb, 
                                                    support_gold_span_tag, query_gold_spans, query_gold_span_tag,
                                                    query_indices, query_label_emb)
        
            cur_support_type_loss = self.Type_Loss_V0(support_span_scores.detach(), support_bert_emb, 
                                                    support_gold_span_tag, support_gold_spans, support_gold_span_tag,
                                                    support_indices, support_label_emb)
            
            bs += 1

            loss = loss + (cur_support_type_loss + cur_query_type_loss)  + (cur_support_gold_loss + cur_query_gold_loss) 
        
            Type_loss += (cur_support_type_loss + cur_query_type_loss) 
            support_CL_loss += cur_support_gold_loss
            query_CL_loss += cur_query_gold_loss
            support_gold_num += support_cur_gold_num
            query_gold_num += query_cur_gold_num
            support_num += support_cur_num
            query_num += query_cur_num

        loss /= bs
        Type_loss /= bs
        support_CL_loss /= bs
        query_CL_loss /= bs
        support_gold_loss /= bs
        query_gold_loss /= bs
    

        return {'loss': loss, "Type_loss": Type_loss, "Span_loss":Span_loss, "support_CL_loss": support_CL_loss, "query_CL_loss": query_CL_loss,
                "CL_loss": support_CL_loss + query_CL_loss, "support_gold_loss": support_gold_loss,
                "query_gold_loss": query_gold_loss, "Second_stage_loss": Type_loss + support_CL_loss + query_CL_loss}
                
    def train_step_stage_two(self, support, query):
        device = support['prefix_prompt_word'].device

        support_num = query_num = 0
        support_gold_num = query_gold_num = 0


        loss = 0
        support_CL_loss = 0
        query_CL_loss = 0
        support_gold_loss = 0
        query_gold_loss = 0
        Type_loss = 0
        bs = 0

        for support_cur_num, query_cur_num in zip(support['sentence_num'], query['sentence_num']):
            
            
            support_out_1 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768  
            
            query_out_1 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                           output_hidden_states=True,
                                           return_dict=True)  # num_sent x number_of_tokens x 768
        
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])
            query_cur_seq_len = max(query['seq_len'][query_num: query_num + query_cur_num])
            query_cur_prefix_seq_len = max(query['prefix_seq_len'][query_num: query_num + query_cur_num])

            support_emb_1 = support_out_1['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]

            query_emb_1 = query_out_1['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]


            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            query_indices = query['indices'][query_num:query_num + query_cur_num][:, 0:query_cur_seq_len]
     
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]
            query_prefix_indices = query['prefix_indices'][query_num:query_num + query_cur_num][:, 0:query_cur_prefix_seq_len]

            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            query_gold_spans = query['gold_spans'][query_num:query_num + query_cur_num]

            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            query_gold_span_num = query['gold_span_num'][query_num:query_num + query_cur_num]
            query_cur_gold_num = sum(query_gold_span_num)

            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)
            query_gold_span_tag = query['gold_span_tag'][query_gold_num: query_gold_num + query_cur_gold_num].to(device)


            self.N = support_gold_span_tag.max()

            support_label_emb = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, 1:self.N + 2]
            support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        
            query_label_emb = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, 1:self.N + 2]
            query_emb_1 = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

            support_bert_emb = support_emb_1
            query_bert_emb = query_emb_1
            
            if self.K == 5:
                support_bert_emb_CL = support_emb_1
                query_bert_emb_CL = query_emb_1
            else: 
                support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768              

                support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]   # bsz x seq_len x hidden_size
                support_bert_emb_CL = support_emb_2

                query_out_2 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768    
                query_emb_2 = query_out_2['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
                query_emb_2 = scatter_mean(query_emb_2, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

                query_bert_emb_CL = query_emb_2

            support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            support_gold_rep_CL = self.get_gold_span_rep(support_bert_emb_CL, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            
            query_gold_rep = self.get_gold_span_rep(query_bert_emb, query_gold_spans, query_gold_span_tag)  # gold_num x 768
            query_gold_rep_CL = self.get_gold_span_rep(query_bert_emb_CL, query_gold_spans, query_gold_span_tag)  # gold_num x 768

            cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep_CL, support_gold_span_tag)

            cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep_CL, query_gold_span_tag)
            
            cur_support_type_loss = self.Type_Loss(support_bert_emb, support_gold_span_tag, support_gold_spans, support_gold_span_tag,support_indices, support_label_emb)
            cur_query_type_loss = self.Type_Loss(query_bert_emb, support_gold_span_tag, query_gold_spans, query_gold_span_tag, query_indices, query_label_emb)
            
            bs += 1

            loss = loss + (cur_support_type_loss + cur_query_type_loss) # + (cur_support_gold_loss + cur_query_gold_loss) 
        
            Type_loss += (cur_support_type_loss + cur_query_type_loss) 
            support_CL_loss += cur_support_gold_loss
            query_CL_loss += cur_query_gold_loss
            support_gold_num += support_cur_gold_num
            query_gold_num += query_cur_gold_num
            support_num += support_cur_num
            query_num += query_cur_num

        loss /= bs
        Type_loss /= bs
        support_CL_loss /= bs
        query_CL_loss /= bs
        support_gold_loss /= bs
        query_gold_loss /= bs
    

        return {'loss': loss, "Type_loss": Type_loss, "Span_loss": 0.0000, "support_CL_loss": support_CL_loss, "query_CL_loss": query_CL_loss,
                "CL_loss": support_CL_loss + query_CL_loss, "support_gold_loss": support_gold_loss,
                "query_gold_loss": query_gold_loss, "Second_stage_loss": Type_loss + support_CL_loss + query_CL_loss}
    
    def train_step_stage_two_V1(self, support, query):
        device = support['prefix_prompt_word'].device

        support_num = query_num = 0
        support_gold_num = query_gold_num = 0

        support_max_seq_len, query_max_seq_len = max(support['seq_len']), max(query['seq_len'])

        loss = 0
        support_CL_loss = 0
        query_CL_loss = 0
        support_gold_loss = 0
        query_gold_loss = 0
        Type_loss = 0
        Span_loss = 0
        bs = 0

        for support_cur_num, query_cur_num in zip(support['sentence_num'], query['sentence_num']):
            support_attention_mask = seq_len_to_mask(torch.tensor(support['seq_len'][support_num:
                                                                                     support_num + support_cur_num]),
                                                     max_len=support_max_seq_len).int().to(device)
            """
            support_out = self.pretrain_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            """
            support_out = self.pretrain_model(support['word'][support_num: support_num + support_cur_num],
                                              attention_mask=support_attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
            
            
            support_out_1 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768  

            query_attention_mask = seq_len_to_mask(torch.tensor(query['seq_len'][query_num:query_num + query_cur_num]),
                                                   max_len=query_max_seq_len).int().to(device)
            """
            query_out = self.pretrain_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            """
            query_out = self.pretrain_model(query['word'][query_num: query_num + query_cur_num],
                                            attention_mask=query_attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)  # num_sent x number_of_tokens x 768
            
            query_out_1 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                           output_hidden_states=True,
                                           return_dict=True)  # num_sent x number_of_tokens x 768
        
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])
            query_cur_seq_len = max(query['seq_len'][query_num: query_num + query_cur_num])
            query_cur_prefix_seq_len = max(query['prefix_seq_len'][query_num: query_num + query_cur_num])

            support_emb = support_out['last_hidden_state'][:, 0:support_cur_seq_len, :]
            #support_emb = support_out['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
            support_emb_1 = support_out_1['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]

            query_emb = query_out['last_hidden_state'][:, 0:query_cur_seq_len, :]
            #query_emb = query_out['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
            query_emb_1 = query_out_1['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]


            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            query_indices = query['indices'][query_num:query_num + query_cur_num][:, 0:query_cur_seq_len]
     
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]
            query_prefix_indices = query['prefix_indices'][query_num:query_num + query_cur_num][:, 0:query_cur_prefix_seq_len]

            support_emb = scatter_mean(support_emb, index=support_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            #support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2::]  # bsz x seq_len x hidden_size
            support_label_emb = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, 1:self.N + 2]
            support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        
            query_emb = scatter_mean(query_emb, index=query_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            #query_emb = scatter_mean(query_emb, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
            query_label_emb = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, 1:self.N + 2]
            query_emb_1 = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

            support_bert_emb = support_emb_1
            query_bert_emb = query_emb_1
            
            if self.K == 5:
                support_bert_emb_CL = support_emb_1
                query_bert_emb_CL = query_emb_1
            else: 
                support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768              

                support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]   # bsz x seq_len x hidden_size
                support_bert_emb_CL = support_emb_2

                query_out_2 = self.proto_model(query['prefix_prompt_word'][query_num: query_num + query_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768    
                query_emb_2 = query_out_2['last_hidden_state'][:, 0:query_cur_prefix_seq_len, :]
                query_emb_2 = scatter_mean(query_emb_2, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

                query_bert_emb_CL = query_emb_2

            # Biaffine Module
            support_span_scores = self(support_emb, support_indices)  # bsz x seq_len x seq_len x dim
            query_span_scores = self(query_emb, query_indices)  # bsz x seq_len x seq_len x dim

            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            query_gold_spans = query['gold_spans'][query_num:query_num + query_cur_num]

            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            query_gold_span_num = query['gold_span_num'][query_num:query_num + query_cur_num]
            query_cur_gold_num = sum(query_gold_span_num)

            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)
            query_gold_span_tag = query['gold_span_tag'][query_gold_num: query_gold_num + query_cur_gold_num].to(device)

            support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            support_gold_rep_CL = self.get_gold_span_rep(support_bert_emb_CL, support_gold_spans, support_gold_span_tag)  # gold_num x 768

            query_gold_rep = self.get_gold_span_rep(query_bert_emb, query_gold_spans, query_gold_span_tag)  # gold_num x 768
            query_gold_rep_CL = self.get_gold_span_rep(query_bert_emb_CL, query_gold_spans, query_gold_span_tag)  # gold_num x 768

            cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep_CL, support_gold_span_tag)
            #cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep, support_gold_span_tag)

            cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep_CL, query_gold_span_tag)
            #cur_query_gold_loss = self.get_gold_CL_loss(query_gold_rep, query_gold_rep, query_gold_span_tag)
            #cur_query_type_loss = self.Type_Loss(query_span_scores.detach(), query_bert_emb, 
                                                    #support_gold_span_tag, query_gold_spans, query_gold_span_tag,
                                                   # query_indices, query_label_emb)
            cur_query_type_loss = self.Type_Loss_V1(query_span_scores.detach(), query_bert_emb, support_gold_rep, support_gold_span_tag, query_gold_spans, query_gold_span_tag, query_indices, query_label_emb)
            cur_support_type_loss = self.Type_Loss_V1(support_span_scores.detach(), support_bert_emb, query_gold_rep, query_gold_span_tag, support_gold_spans, support_gold_span_tag, support_indices, support_label_emb)
           # cur_support_type_loss = self.Type_Loss(support_span_scores.detach(), support_bert_emb, 
                                                #    support_gold_span_tag, support_gold_spans, support_gold_span_tag,
                                                   # support_indices, support_label_emb)
            
            bs += 1

            loss = loss + (cur_support_type_loss + cur_query_type_loss)  + (cur_support_gold_loss + cur_query_gold_loss) 
            #loss = loss +  cur_query_type_loss + (support_span_loss + query_span_loss) / 2  + (cur_support_gold_loss + cur_query_gold_loss) / 2
            Type_loss += (cur_support_type_loss + cur_query_type_loss) 
            #Type_loss = cur_query_type_loss

            support_CL_loss += cur_support_gold_loss
            query_CL_loss += cur_query_gold_loss

            support_gold_num += support_cur_gold_num
            query_gold_num += query_cur_gold_num
            support_num += support_cur_num
            query_num += query_cur_num

        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        support_CL_loss /= bs
        query_CL_loss /= bs
        support_gold_loss /= bs
        query_gold_loss /= bs
    

        return {'loss': loss, "Type_loss": Type_loss, "Span_loss":Span_loss, "support_CL_loss": support_CL_loss, "query_CL_loss": query_CL_loss,
                "CL_loss": support_CL_loss + query_CL_loss, "support_gold_loss": support_gold_loss,
                "query_gold_loss": query_gold_loss, "Second_stage_loss": Type_loss + support_CL_loss + query_CL_loss}
    
    def train_step(self, batch):
        device = batch['word'].device

        batch_num = 0
        batch_gold_num =  0

        support_max_seq_len = max(batch['seq_len'])

        loss = 0
        batch_CL_loss = 0

        batch_gold_loss = 0
  
        Type_loss = 0
        Span_loss = 0
        bs = 0

        for batch_cur_num in batch['sentence_num']:
            support_attention_mask = seq_len_to_mask(torch.tensor(batch['seq_len'][batch_num: batch_num + batch_cur_num]),
                                                     max_len=support_max_seq_len).int().to(device)

            batch_out = self.pretrain_model(batch['word'][batch_num: batch_num + batch_cur_num],
                                              attention_mask=support_attention_mask,
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768

            batch_out_1 = self.proto_model(batch['prefix_prompt_word'][batch_num: batch_num + batch_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768  


        
            batch_cur_seq_len = max(batch['seq_len'][batch_num: batch_num + batch_cur_num])
            batch_cur_prefix_seq_len = max(batch['prefix_seq_len'][batch_num: batch_num + batch_cur_num])


            batch_emb = batch_out['last_hidden_state'][:, 0:batch_cur_seq_len, :]
            batch_emb_1 = batch_out_1['last_hidden_state'][:, 0:batch_cur_prefix_seq_len, :]

            # torch_scatter
            batch_indices = batch['indices'][batch_num:batch_num + batch_cur_num][:, 0:batch_cur_seq_len]
         
     
            batch_prefix_indices = batch['prefix_indices'][batch_num:batch_num + batch_cur_num][:, 0:batch_cur_prefix_seq_len]
         

            batch_emb = scatter_mean(batch_emb, index=batch_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            batch_label_emb = scatter_mean(batch_emb_1, index=batch_prefix_indices, dim=1)[:, 1:self.N + 2]
            batch_emb_1 = scatter_mean(batch_emb_1, index=batch_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        
            batch_bert_emb = batch_emb_1

 
            
            if self.K == 5:
                batch_bert_emb_CL = batch_emb_1
            else: 
                batch_out_2 = self.proto_model(batch['prefix_prompt_word'][batch_num: batch_num + batch_cur_num],
                                                output_hidden_states=True,
                                                return_dict=True)  # num_sent x number_of_tokens x 768              

                batch_emb_2 = batch_out_2['last_hidden_state'][:, 0:batch_cur_prefix_seq_len, :]
                batch_emb_2 = scatter_mean(batch_emb_2, index=batch_prefix_indices, dim=1)[:, self.N + 2:]   # bsz x seq_len x hidden_size
                batch_bert_emb_CL = batch_emb_2

            # Biaffine Module
            batch_span_scores = self(batch_emb, batch_indices)  # bsz x seq_len x seq_len x dim

            batch_gold_spans = batch['gold_spans'][batch_num: batch_num + batch_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list


            batch_gold_span_num = batch['gold_span_num'][batch_num: batch_num + batch_cur_num]
            batch_cur_gold_num = sum(batch_gold_span_num)

            batch_gold_span_tag = batch['gold_span_tag'][batch_gold_num: batch_gold_num + batch_cur_gold_num].to(device)


            batch_span_loss = self.Span_Loss(batch_span_scores, batch_gold_spans, batch_indices)
    

            batch_gold_rep = self.get_gold_span_rep(batch_bert_emb, batch_gold_spans, batch_gold_span_tag)  # gold_num x 768
            batch_gold_rep_CL = self.get_gold_span_rep(batch_bert_emb_CL, batch_gold_spans, batch_gold_span_tag)  # gold_num x 768
            
            if self.N == 10 and self.K == 5:
                cur_batch_gold_loss = 0
            else:
                cur_batch_gold_loss = self.get_gold_CL_loss(batch_gold_rep, batch_gold_rep_CL, batch_gold_span_tag)

            cur_batch_type_loss = self.Type_Loss(batch_span_scores.detach(), batch_bert_emb, 
                                                    batch_gold_span_tag, batch_gold_spans, batch_gold_span_tag,
                                                    batch_indices, batch_label_emb)

            bs += 1
           
            loss = loss + cur_batch_type_loss + batch_span_loss  + cur_batch_gold_loss 
 
            Type_loss += cur_batch_type_loss 
            #Type_loss = cur_query_type_loss
            Span_loss += batch_span_loss

            batch_gold_loss += batch_span_loss  # cur_support_gold_loss
   
            batch_CL_loss += cur_batch_gold_loss


            batch_gold_num += batch_cur_gold_num

            batch_num += batch_cur_num


        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        batch_CL_loss /= bs
        batch_gold_loss /= bs
      
        return {'loss': loss, "Type_loss": Type_loss, "Span_loss":Span_loss, "support_CL_loss": batch_CL_loss, "query_CL_loss": batch_CL_loss,
                "CL_loss": batch_CL_loss, "support_gold_loss": batch_gold_loss,
                "query_gold_loss": batch_gold_loss, "Second_stage_loss": Type_loss + batch_CL_loss}

    def get_gold_CL_loss(self, gold_rep, gold_rep_CL=None, gold_span_tag=None, notQueryInSnips=True):
        device = gold_rep.device
        scores = torch.einsum("xy, zy->xz", gold_rep, gold_rep_CL) / (768 ** 0.5)

        gold_num = gold_span_tag.size(0)
        
        tag_1 = torch.repeat_interleave(gold_span_tag, gold_num, dim=-1)
        tag_2 = gold_span_tag.repeat(1, gold_num)

        pos_mask = (tag_1 == tag_2).int().view(gold_num, -1).to(device)
        scores = torch.exp(scores)
        if self.K == 5 and notQueryInSnips:
            sum_mask = 1 - torch.diag_embed(torch.ones(gold_num)).to(device)
            pos_mask = pos_mask * sum_mask
            sum_scores = torch.sum(scores * sum_mask, dim=-1)
        else:
            sum_scores = torch.sum(scores, dim=-1)

        pos_scores = torch.sum(pos_mask * scores, dim=-1)
    
        scores = torch.log(pos_scores / sum_scores)
        if self.K == 1:
            loss = -torch.sum(scores)
        else:
            loss = -torch.mean(scores)
        return loss 

    def fine_tuning_step(self, support):
        device = support['prefix_prompt_word'].device

        support_num = 0
        support_gold_num = 0

        loss = 0
        support_CL_loss = 0

        Span_loss = 0
        Type_loss = 0
        bs = 0

        for support_cur_num in support['sentence_num']:
            support_out = self.pretrain_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768

            support_out_1 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768
           
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])

            #support_emb = support_out['last_hidden_state'][:, 0:support_cur_seq_len, :]
            support_emb = support_out['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
            support_emb_1 = support_out_1['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
 
            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]

            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)

            self.N = support_gold_span_tag.max()

            #support_emb = scatter_mean(support_emb, index=support_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
            support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
            support_label_emb = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, 1:self.N + 2]
            support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size


            support_bert_emb = support_emb_1
            if self.K == 5:
                support_bert_emb_CL = support_emb_1
            else:
                support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768                                             
                support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
                support_bert_emb_CL = support_emb_2
            # Biaffine + CNN Module
            support_span_scores = self(support_emb, support_indices)  # bsz x seq_len x seq_len x dim

            support_span_loss = self.Span_Loss_V2(support_span_scores, support_gold_spans, support_indices)
           
            support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            if self.K == 5:
                support_gold_rep_CL = support_gold_rep
            else:
                support_gold_rep_CL = self.get_gold_span_rep(support_bert_emb_CL, support_gold_spans, support_gold_span_tag)  # gold_num x 768

            cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep_CL, support_gold_span_tag)

            support_type_loss = self.Type_Loss(support_bert_emb, support_gold_span_tag, support_gold_spans, support_gold_span_tag, support_indices, support_label_emb)

            bs += 1

            loss = loss + support_span_loss + support_type_loss #+ cur_support_gold_loss

            Type_loss += support_type_loss
            Span_loss += support_span_loss 
            support_gold_num += support_cur_gold_num
            support_CL_loss += cur_support_gold_loss
            support_num += support_cur_num

        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        support_CL_loss /= bs

        return {'loss': loss, "Span_loss": Span_loss, "support_CL_loss": support_CL_loss, "CL_loss": support_CL_loss}
 
    def Type_Loss_V0(self, logits, query_bert_emb, support_gold_span_tag, gold_spans, gold_span_tag, indices, label_emb):
        device = logits.device
        bsz, seq_len, seq_len = logits.size()
        num_class = support_gold_span_tag.max()
        lengths, _ = indices.max(dim=-1)
        y_type_target = torch.ones(bsz, seq_len, seq_len, dtype=torch.long).to(device) * (-100)
        #dists = torch.zeros(bsz, seq_len, seq_len, num_class+1).to(device)
        dists_label = torch.zeros(bsz, seq_len, seq_len, num_class+1).to(device)
        num = 0
        if self.mode == 'inter11' and self.N != 10:# and self.K != 5:
            for length, logit in zip(lengths, logits):        
                scores = torch.sigmoid(logit[:length, :length])
        
                random = torch.rand(scores.shape).to(device)
                random_scores = torch.triu(random, diagonal=0)
            
                mask = torch.triu(torch.ones(scores.shape).to(device), diagonal=0) # 上三角及对角线
                scores = scores * mask

                indices = self.get_TopK_span(random_scores, 5, 0.0)
    
                if indices[0].size(0) > 0:
                    span_rep = self.get_span_rep(query_bert_emb[num], indices) # span_num x bert_hsz
            
                    label_dists = (span_rep.unsqueeze(1) * label_emb[num].unsqueeze(0)).sum(dim=-1) / (768 ** 0.5) # span_num x num_class

                    #span_num = label_dists.size(0)
                    #O_dists = torch.zeros(span_num, 1).to(device)
                    #label_dists = torch.cat([O_dists, label_dists], dim=-1).to(device)
                    label_dists = F.softmax(label_dists, dim=-1) 

                    dists_label[num, indices[0], indices[1], :] = label_dists
 
                    y_type_target[num, indices[0], indices[1]] = 0
                num += 1
            
        gold_num = 0

        for idx in range(bsz):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue
            gold_span = torch.tensor(gold_span).to(device)
            gold_cur_num = gold_span.size(0)
            indices = (gold_span[:, 0], gold_span[:, 1])

            span_rep = self.get_span_rep(query_bert_emb[idx], indices)

            tag = gold_span_tag[gold_num: gold_num + gold_cur_num]

            gold_num += gold_cur_num
            #cur_label_emb = torch.cat([label_emb[idx], label_emb[idx], label_emb[idx]], dim=-1)
            #label_dists = (span_rep.unsqueeze(1) * cur_label_emb.unsqueeze(0)).sum(dim=-1) / ((768 * 3) ** 0.5) # span_num x num_class
            label_dists = (span_rep.unsqueeze(1) * label_emb[idx].unsqueeze(0)).sum(dim=-1) / (768 ** 0.5) # span_num x num_class
            dists_label[idx, indices[0], indices[1], :] = label_dists

            y_type_target[idx, gold_span[:, 0], gold_span[:, 1]] = tag  # tag = 0 means O_type

        y_type_target = y_type_target.reshape(-1)
        y_type_pred = dists_label.reshape(-1, num_class+1)
        label_type_loss = F.cross_entropy(y_type_pred, y_type_target, ignore_index=-100, reduction='mean')

        return label_type_loss

    def Type_Loss(self, query_bert_emb, support_gold_span_tag, gold_spans, gold_span_tag, indices, label_emb):
        if gold_span_tag.size(0) == 0:
            return 0
        device = query_bert_emb.device
        bsz, seq_len, _ = query_bert_emb.size()
        num_class = support_gold_span_tag.max()

        y_type_target = torch.ones(bsz, seq_len, seq_len, dtype=torch.long).to(device) * (-100)
        dists_label = torch.zeros(bsz, seq_len, seq_len, num_class+1).to(device)

        gold_num = 0

        for idx in range(bsz):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue
            gold_span = torch.tensor(gold_span).to(device)
            gold_cur_num = gold_span.size(0)
            indices = (gold_span[:, 0], gold_span[:, 1])

            span_rep = self.get_span_rep(query_bert_emb[idx], indices)

            tag = gold_span_tag[gold_num: gold_num + gold_cur_num]

            gold_num += gold_cur_num

            label_dists = (span_rep.unsqueeze(1) * label_emb[idx].unsqueeze(0)).sum(dim=-1) / (768 ** 0.5) # span_num x (num_class+1)

            dists_label[idx, indices[0], indices[1], :] = label_dists

            y_type_target[idx, gold_span[:, 0], gold_span[:, 1]] = tag  # tag = 0 means none - entity

        y_type_target = y_type_target.reshape(-1)
        y_type_pred = dists_label.reshape(-1, num_class+1)
        label_type_loss = F.cross_entropy(y_type_pred, y_type_target, ignore_index=-100, reduction='mean')

        return label_type_loss

    def evaluate_step_NN(self, support, query):
        device = support['prefix_prompt_word'].device

        support_attention_mask = seq_len_to_mask(seq_len=torch.tensor(support['seq_len'])).to(device)
       
        support_out = self.pretrain_model(support['prefix_prompt_word'],
                                          output_hidden_states=True,
                                          return_dict=True)  # num_sent x number_of_tokens x 768
        """
        support_out = self.pretrain_model(support['word'],
                                          attention_mask=support_attention_mask,
                                          output_hidden_states=True,
                                          return_dict=True)  # num_sent x number_of_tokens x 768
        """
        support_out_1 = self.proto_model(support['prefix_prompt_word'],
                                         output_hidden_states=True,
                                         return_dict=True)   # num_sent x number_of_tokens x 768

        query_attention_mask = seq_len_to_mask(seq_len=torch.tensor(query['seq_len'])).to(device)
        
        query_out = self.pretrain_model(query['prefix_prompt_word'],
                                        output_hidden_states=True,
                                        return_dict=True)  # num_sent x number_of_tokens x 768
        """
        query_out = self.pretrain_model(query['word'],
                                        attention_mask=query_attention_mask,
                                        output_hidden_states=True,
                                        return_dict=True)  # num_sent x number_of_tokens x 768
        """
        query_out_1 = self.proto_model(query['prefix_prompt_word'],
                                       output_hidden_states=True,
                                       return_dict=True)  # num_sent x number_of_tokens x 768

        support_emb = support_out['last_hidden_state']
        support_emb_1 = support_out_1['last_hidden_state']
        query_emb = query_out['last_hidden_state']
        query_emb_1 = query_out_1['last_hidden_state']

        # torch_scatter
        support_indices = support['indices']
        support_prefix_indices = support['prefix_indices']
        query_indices = query['indices']
        query_prefix_indices = query['prefix_indices']

        support_gold_span_tag = support['gold_span_tag'].to(device)
        self.N = support_gold_span_tag.max()

        #support_emb = scatter_mean(support_emb, index=support_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
        support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        
        #query_emb = scatter_mean(query_emb, index=query_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
        query_emb = scatter_mean(query_emb, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
        #query_emb_1 = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, 1:]  # bsz x seq_len x hidden_size
        query_label_emb = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, 1:self.N + 2]
        query_emb_1 = scatter_mean(query_emb_1, index=query_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

        # for typing
        support_bert_emb = support_emb_1
        query_bert_emb = query_emb_1
   
        # Biaffine + CNN Module
        query_span_logits = self(query_emb, query_indices)  # bsz x seq_len x seq_len x dim
        
        support_gold_spans = support['gold_spans']
        query_gold_spans = query['gold_spans']

        support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768

        #query_gold_rep = self.get_gold_span_rep(query_bert_emb, query_gold_spans, query_gold_span_tag)
        #rep = torch.cat([support_gold_rep, query_gold_rep], dim=0)
        #tag = torch.cat([support_gold_span_tag, query_gold_span_tag], dim=0)
        #self.tSNE_plt(rep, tag)
        
        score, span, pred = self.predict_two_stage_NN_Label(query_span_logits, query_bert_emb, support_gold_rep,
                                                            support_gold_span_tag, query_indices, self.predict_threshold,
                                                            query_label_emb)

        return {'score': score, 'span': span, 'pred': pred}

    def predict_two_stage_NN_Label(self, logits, query_bert_emb, gold_rep, support_gold_span_tag, indices, threshold, label_emb):
        """
        Args:
            logits: bsz x seq_len x seq_len  是否属于一个span (二分类)
            query_bert_emb:  bsz x seq_len x hidden_size
            gold_rep: gold_num x hidden_size
            support_gold_span_tag: gold_span_num
            indices: bsz x seq_len
            threshold: 一个span是否属于某个label，二分类的阈值
        Returns:
            score_list: bsz x span_num_of_each_sent (tensor lists without padding)
            span_list: bsz x span_num_of_each_sent x 2 (begin, end) (tensor lists without padding)
            pred_list: bsz x span_num_of_each_sent (tensor lists without padding)
        """
        device = logits.device
        score_list, span_list, pred_list = list(), list(), list()
        lengths, _ = indices.max(dim=-1)
        num_class = support_gold_span_tag.max()
        idx = 0


        for length, logit in zip(lengths, logits):

            score_matrix = torch.sigmoid(logit[:length, :length])
            scores = score_matrix
            mask = torch.triu(torch.ones(scores.shape).to(device), diagonal=0)
            scores = scores * mask

            cur_threshold = threshold
            #self.predict_num = ((1 + length) * length // 2)
            #indices = self.get_TopK_span(scores, k=self.predict_num, threshold=0)
            indices = self.get_TopK_span(scores, k=self.predict_num, threshold=cur_threshold)
            while len(indices[0]) == 0 and cur_threshold >= 0.005:
                cur_threshold -= 0.005
                indices = self.get_TopK_span(scores, k=self.predict_num, threshold=cur_threshold)
           
            span = torch.stack([indices[0], indices[1]], dim=-1).to(device)  # span_num_of_each_sent x 2
 
            span_num = span.size(0)
            
            if span_num == 0:
                score = torch.tensor([]).to(device)
                pred = torch.tensor([]).to(device)
            else:
                query_span_rep = self.get_span_rep(query_bert_emb[idx], indices)  # span_num_of_each_sent x dim

                type_dists = (query_span_rep.unsqueeze(1) * gold_rep.unsqueeze(0)).sum(dim=-1) / (768 ** 0.5)  # span_num_of_each_sent x gold_num
                gold_num = gold_rep.size(0)
                # topk_values, topk_indices = torch.topk(type_dists, dim=-1, k=min(self.K * 2, gold_num // 2))
                topk_values, topk_indices = torch.topk(type_dists, dim=-1, k=min(self.K * 2, int(gold_num / self.kNN_ratio)))
                # topk_values, topk_indices = torch.topk(type_dists, dim=-1, k=gold_num)
                topk_values = torch.softmax(topk_values, dim=-1)

                topk_class = support_gold_span_tag[topk_indices]

                label_dists = (query_span_rep.unsqueeze(1) * label_emb[idx].unsqueeze(0)).sum(dim=-1) / (768 ** 0.5) # span_num x （num_class + 1）

                type_dists = torch.softmax(label_dists, dim=-1) 
          
                _, labels = type_dists.max(dim=-1)
 
                none_indices = torch.where(labels == 0)[0]

                type_dists = type_dists * 0.35 # 如果topk没有找出某个类别的entity，那么对应knn概率为0

                for id in range(span_num):
                    type_dists[id, topk_class[id]] = type_dists[id, topk_class[id]] + topk_values[id] * 0.65
                            
                type_dists = self.beta * type_dists + (1-self.beta) * scores[indices[0], indices[1]].unsqueeze(1) # rerank

                scores, labels = type_dists.max(dim=-1)

                pred = labels
                score = scores

                non_entity_indices = torch.where(pred == 0)[0]

                if non_entity_indices.size(0) > 0:
                    score[non_entity_indices] = 0.0
                if none_indices.size(0) > 0:
                    score[none_indices] = 0.0
            score_list.append(score.cpu().numpy().tolist())
            span_list.append(span.cpu().numpy().tolist())
            pred_list.append(pred.cpu().numpy().tolist())
            idx += 1
        return score_list, span_list, pred_list

    def Span_Loss(self, logits, gold_spans, indices):
        """
        Softplus形式的loss
        Args:
            logits: bsz x seq_len x seq_len (二分类判断一个span是否是一个entity）
            gold_spans: bsz x gold_num x 2 的 list
                        e.g. [[[10, 12]], [[10, 11]], [[10, 10], [13, 13]], [[6, 8]], [[3, 4]]]
            indices: bsz x seq_len
        """
        device = logits.device
        lengths, _ = indices.max(dim=-1)
        sigma = torch.zeros(logits.size()).to(device)
        bsz, seq_len, seq_len = logits.size()

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long).to(device), diagonal=-1)  # 下三角
        mask_indices = torch.where(mask == 1)
        
        for idx in range(bsz):
            length = lengths[idx]
            sigma[idx][:length, :length] = 1  # for a non-entity span， sigma[i,j] = 1
            sigma[idx][mask_indices] = 0  # for an entry which satisfies j > i

        gold_num = 0
        for idx in range(bsz):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue
            gold_span = torch.tensor(gold_span).to(device)
            gold_cur_num = gold_span.size(0)
            gold_num += gold_cur_num
            sigma[idx][gold_span[:, 0], gold_span[:, 1]] = -1  # for an entity， sigma[i,j] = -1
        
        span_loss = 0
        for idx in range(bsz):
            cur_sigma = sigma[idx].reshape(-1)
            y_pred = logits[idx].reshape(-1)

            mask = cur_sigma.ne(0).int().to(device)
            exp_scores = torch.exp(y_pred * cur_sigma)
            scores = (mask * exp_scores).sum()
            span_loss += torch.log(1 + scores)
           
        return span_loss / bsz

    def Span_Loss_V1(self, logits, gold_spans, indices):
        """
        Softplus形式的loss
        Args:
            logits: bsz x seq_len x seq_len (二分类判断一个span是否是一个entity）
            gold_spans: bsz x gold_num x 2 的 list
                        e.g. [[[10, 12]], [[10, 11]], [[10, 10], [13, 13]], [[6, 8]], [[3, 4]]]
            indices: bsz x seq_len
        """
        device = logits.device
        lengths, _ = indices.max(dim=-1)
        sigma = torch.zeros(logits.size()).to(device)
        bsz, seq_len, seq_len = logits.size()

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long).to(device), diagonal=-1)  # L x L矩阵下三角不参与算Loss
        mask_indices = torch.where(mask == 1)
        
        for idx in range(bsz):
            length = lengths[idx]
            sigma[idx][:length, :length] = 1  # for a non-entity span， sigma[i,j] = 1
            sigma[idx][mask_indices] = 0  # for an entry which satisfies j > i

        gold_num = 0
        for idx in range(bsz):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue
            gold_span = torch.tensor(gold_span).to(device)
            gold_cur_num = gold_span.size(0)
            gold_num += gold_cur_num
            sigma[idx][gold_span[:, 0], gold_span[:, 1]] = -1  # for an entity， sigma[i,j] = -1
            


        span_loss = 0
        span_positive_loss = 0
        span_negative_loss = 0
        for idx in range(bsz):
            cur_sigma = sigma[idx].reshape(-1)
            y_pred = logits[idx].reshape(-1)

            positive_mask = cur_sigma.eq(-1).int().to(device)
            negative_mask = cur_sigma.eq(1).int().to(device)

    
            exp_scores = torch.exp(y_pred * cur_sigma)
            positive_scores = (positive_mask * exp_scores).sum()
            negative_scores = (negative_mask * exp_scores).sum()

            span_positive_loss = torch.log(1 + positive_scores)
            span_negative_loss = torch.log(1 + negative_scores)

            span_loss += (span_negative_loss + span_positive_loss)
        return span_loss / bsz
    
    def Span_Loss_V2(self, logits, gold_spans, indices):
        """
        follow globalpointer's loss function
        Args:
            logits: bsz x seq_len x seq_len (二分类判断一个span是否是一个entity）
            gold_spans: bsz x gold_num x 2 的 list
                        e.g. [[[10, 12]], [[10, 11]], [[10, 10], [13, 13]], [[6, 8]], [[3, 4]]]
            indices: bsz x seq_len
        """
        device = logits.device
        lengths, _ = indices.max(dim=-1)
        sigma = torch.zeros(logits.size()).to(device)
        bsz, seq_len, seq_len = logits.size()

        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long).to(device), diagonal=-1)  # loss_mask
        mask_indices = torch.where(mask == 1)
        
        mask = torch.zeros(logits.size()).to(device)

        for idx in range(bsz):
            length = lengths[idx]
            mask[idx][:length, :length] = 1
        mask = torch.triu(mask) 
        y_true = torch.zeros(logits.size()).to(device)

        gold_num = 0
        for idx in range(bsz):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue

            gold_span = torch.tensor(gold_span).to(device)
            y_true[idx][gold_span[:, 0], gold_span[:, 1]] = 1 #
            gold_cur_num = gold_span.size(0)
            gold_num += gold_cur_num
        y_pred = logits - (1 - mask) * 1e12
        y_true = y_true.view(bsz, -1)
        y_pred = y_pred.view(bsz, -1)
        loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
        
        return loss

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def get_TopK_span(self, scores, k, threshold):
   
        scores = torch.triu(scores, diagonal=0)
        length = scores.size(0)
        scores = scores.reshape(-1)

        topk = torch.topk(scores, k=min(k, length), dim=-1)
        indices = topk.indices
        values = topk.values
        idx = 0
        num = values.size(0)
        for value in values:
            if value >= threshold:
                idx += 1
            else:
                break
        
 
        begin = torch.div(indices, length, rounding_mode='trunc')
        end = indices % length

        return (begin[:idx], end[:idx])

    def get_gold_span_rep(self, embedding, gold_spans, gold_span_tag):
        """
        得到gold span的rep
        Args:
            embedding:  bsz x seq_len x bert_dim
            gold_spans: bsz x gold_num x 2 的 list
                        e.g [[[10, 12]], [[10, 11]], [[10, 10], [13, 13]], [[6, 8]], [[3, 4]]]
            gold_span_tag: e.g tensor([3, 5, 2, 2, 1, 4])
        Returns:
            type_num x dim
                e.g torch.Size([6, 768]) 按照gold_span_tag的顺序排好
        """
        proto_list = list()
        device = embedding.device

        for idx, emb in enumerate(embedding):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue
            for span in gold_span:
                proto_emb = emb[span[0]:span[1]+1, :].mean(dim=0).view(1, -1)
                proto_list.append(proto_emb)
         
        proto = torch.cat(proto_list, dim=0).to(device)
  
        return proto

    def dot_product(self, query, proto):
        """
        query: [bsz, seq_len, seq_len, span_num, hs]
        proto: [proto_num, hs]

        return: shape  [bsz, seq_len, seq_len, proto_num]
        """
        bsz, seq_len, seq_len, hs = query.size()

        tau = math.sqrt(hs)
        query = query.reshape(-1, hs)

        proto_num, _ = proto.size()

        dist = (query.unsqueeze(1) * proto.unsqueeze(0)).sum(dim=-1)

        return dist.reshape(bsz, seq_len, seq_len, proto_num)

    def cosine_dist(self, query, proto, tau=1.0):
        """
        query: [bsz, seq_len, seq_len, span_num, hs]
        proto: [proto_num, hs]

        return: shape  [bsz, seq_len, seq_len, proto_num]
        """
        bsz, seq_len, seq_len, hs = query.size()
        query = query.reshape(-1, hs)

        proto_num, _ = proto.size()
        dist = F.cosine_similarity(proto.unsqueeze(0), query.unsqueeze(1), dim=-1) / tau

        return dist.reshape(bsz, seq_len, seq_len, proto_num)

    def euclidean_dist(self, query, proto):
        """
        query: [bsz, seq_len, seq_len, hs]
        proto: [proto_num, hs]


        return shape  [span_num, gold_num]
        """
        bsz, seq_len, seq_len, hs = query.size()
        query = query.reshape(-1, hs)
        proto_num, hs = proto.size()
        dist = -((query.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=-1) / hs
        return dist.reshape(bsz, seq_len, seq_len, proto_num)

    def euclidean_dist_NN(self, query, support):
        """
        query: [query_num, hs]
        support: [support_num, hs]

        return shape  [span_num, gold_num]
        """
        _, hs = support.size()
        dist = -((query.unsqueeze(1) - support.unsqueeze(0)) ** 2).sum(dim=-1) / (hs)
  
        return dist

    def cosine_dist_CL(self, x, y, tau=1.0):
        """
        x: [span_num, hs]  sampled spans
        y: [gold_num, hs]  golden spans

        return shape  [span_num, gold_num]
        """
        return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1) / tau
    
    def Type_Loss_V1(self, logits, query_bert_emb, gold_rep, support_gold_span_tag, gold_spans, gold_span_tag, indices, label_emb):
        device = logits.device
        bsz, seq_len, seq_len = logits.size()
        num_class = support_gold_span_tag.max()
        lengths, _ = indices.max(dim=-1)
        y_type_target = torch.ones(bsz, seq_len, seq_len, dtype=torch.long).to(device) * (-100)
        dists = torch.zeros(bsz, seq_len, seq_len, num_class+1).to(device)
        dists_label = torch.zeros(bsz, seq_len, seq_len, num_class+1).to(device)
     
        gold_num = 0

        for idx in range(bsz):
            gold_span = gold_spans[idx]
            if len(gold_span) == 0:
                continue
            gold_span = torch.tensor(gold_span).to(device)
            gold_cur_num = gold_span.size(0)
            indices = (gold_span[:, 0], gold_span[:, 1])

            span_rep = self.get_span_rep(query_bert_emb[idx], indices)

            tag = gold_span_tag[gold_num: gold_num + gold_cur_num]

            gold_num += gold_cur_num
     
            label_dists = (span_rep.unsqueeze(1) * label_emb[idx].unsqueeze(0)).sum(dim=-1) / (768 ** 0.5) # span_num x (num_class + 1)
            span_dists = (span_rep.unsqueeze(1) * gold_rep.unsqueeze(0)).sum(dim=-1) / (768 ** 0.5)
            
            topk_values, topk_indices = torch.topk(span_dists, dim=-1, k=span_dists.size(1))

            topk_values = torch.softmax(topk_values, dim=-1)
     
            topk_class = support_gold_span_tag[topk_indices]

            for id in range(topk_class.size(0)):
                dists[idx, indices[0][id], indices[1][id], topk_class[id]] += topk_values[id] 

            dists_label[idx, indices[0], indices[1], :] = label_dists

            y_type_target[idx, gold_span[:, 0], gold_span[:, 1]] = tag  # tag = 0 means O_type

        y_type_target = y_type_target.reshape(-1)
        y_type_pred = dists_label.reshape(-1, num_class+1)

        label_type_loss = F.cross_entropy(y_type_pred, y_type_target, ignore_index=-100, reduction='mean')
        y_type_pred = dists.reshape(-1, num_class+1)
        type_loss = F.cross_entropy(y_type_pred, y_type_target, ignore_index=-100, reduction='mean')
        return type_loss + label_type_loss 
    
    def fine_tuning_step_stage_one(self, support):
   
        support_num = 0
        support_gold_num = 0

        loss = 0
        Span_loss = 0
        support_CL_loss = 0
        Type_loss = 0
        bs = 0

        for support_cur_num in support['sentence_num']:
            support_out = self.pretrain_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                              output_hidden_states=True,
                                              return_dict=True)  # num_sent x number_of_tokens x 768
           
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])

            support_emb = support_out['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
         
            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]

            support_emb = scatter_mean(support_emb, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size

   
            # Biaffine + CNN Module
            support_span_scores = self(support_emb, support_indices)  # bsz x seq_len x seq_len x dim

            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)

            support_span_loss = self.Span_Loss_V2(support_span_scores, support_gold_spans, support_indices)

            bs += 1

            loss = loss + support_span_loss

            #Type_loss += support_type_loss
            Span_loss += support_span_loss 
            support_gold_num += support_cur_gold_num
            #support_CL_loss += cur_support_gold_loss
            support_num += support_cur_num

        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        support_CL_loss /= bs

        return {'loss': loss, "Span_loss": Span_loss, "Type_loss": Type_loss, "CL_loss": support_CL_loss}
    
    def fine_tuning_step_stage_two(self, support):
        device = support['prefix_prompt_word'].device

        support_num = 0
        support_gold_num = 0

        loss = 0
        support_CL_loss = 0
        Type_loss= 0
        Span_loss = 0
        Type_loss = 0
        bs = 0

        for support_cur_num in support['sentence_num']:

            support_out_1 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768
           
            support_cur_seq_len = max(support['seq_len'][support_num: support_num + support_cur_num])
            support_cur_prefix_seq_len = max(support['prefix_seq_len'][support_num: support_num + support_cur_num])

            support_emb_1 = support_out_1['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
 
            # torch_scatter
            support_indices = support['indices'][support_num:support_num + support_cur_num][:, 0:support_cur_seq_len]
            support_prefix_indices = support['prefix_indices'][support_num:support_num + support_cur_num][:, 0:support_cur_prefix_seq_len]


            support_gold_spans = support['gold_spans'][support_num: support_num + support_cur_num]  # [[[b,e], [b,e]],.....],  bz x gold_num x 2 的 list
            support_gold_span_num = support['gold_span_num'][support_num: support_num + support_cur_num]
            support_cur_gold_num = sum(support_gold_span_num)
            support_gold_span_tag = support['gold_span_tag'][support_gold_num: support_gold_num + support_cur_gold_num].to(device)

            self.N = support_gold_span_tag.max()
            support_emb_1 = scatter_mean(support_emb_1, index=support_prefix_indices, dim=1)
            support_label_emb = support_emb_1[:, 1:self.N + 2]
            support_emb_1 = support_emb_1[:, self.N + 2:]  # bsz x seq_len x hidden_size

            support_bert_emb = support_emb_1
            if self.K == 5:
                support_bert_emb_CL = support_emb_1
            else:
                support_out_2 = self.proto_model(support['prefix_prompt_word'][support_num: support_num + support_cur_num],
                                             output_hidden_states=True,
                                             return_dict=True)  # num_sent x number_of_tokens x 768                                             
                support_emb_2 = support_out_2['last_hidden_state'][:, 0:support_cur_prefix_seq_len, :]
                support_emb_2 = scatter_mean(support_emb_2, index=support_prefix_indices, dim=1)[:, self.N + 2:]  # bsz x seq_len x hidden_size
                support_bert_emb_CL = support_emb_2


            support_gold_rep = self.get_gold_span_rep(support_bert_emb, support_gold_spans, support_gold_span_tag)  # gold_num x 768
            support_gold_rep_CL = self.get_gold_span_rep(support_bert_emb_CL, support_gold_spans, support_gold_span_tag)  # gold_num x 768

            cur_support_gold_loss = self.get_gold_CL_loss(support_gold_rep, support_gold_rep_CL, support_gold_span_tag)
 
            support_type_loss = self.Type_Loss(support_bert_emb, support_gold_span_tag, support_gold_spans, support_gold_span_tag, support_indices, support_label_emb)
           
            bs += 1

            loss = loss + support_type_loss #+ cur_support_gold_loss

            Type_loss += support_type_loss
            support_CL_loss += cur_support_gold_loss
            support_gold_num += support_cur_gold_num
         
            support_num += support_cur_num

        loss /= bs
        Type_loss /= bs
        Span_loss /= bs
        support_CL_loss /= bs

        return {'loss': loss, "Span_loss": Span_loss, "Type_loss": Type_loss, "CL_loss": support_CL_loss}
    
