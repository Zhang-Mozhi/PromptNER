

import torch 
from torch import nn



inputs = torch.ones([4,32,768])


class Biaffine(nn.Module):
    def __init__(self):
        super(Biaffine, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(768, 200),
            nn.ReLU()
        )

        self.tail = nn.Sequential(
            nn.Linear(768, 200),
            nn.ReLU()
        )


        self.dense = nn.Sequential(
            nn.Linear(768, 200),
            nn.ReLU()
        )
        self.biaffine_size = 200

        self.U = nn.Parameter(torch.randn(200, 200, 200))

        torch.nn.init.xavier_normal_(self.U.data)

        self.down_layer = nn.Linear(200, 1)

    def forward(self, inputs):
        
        head = self.head(inputs)
        tail = self.tail(inputs)

        logits = torch.einsum("bxi, oij, byj->boxy", head, self.U, tail).permute(0,2,3,1)
        print(logits.size())
        outputs = self.dense(inputs)


        # rope_logits = self.RoPE(dense)
        outputs = torch.split(outputs, 200, dim=-1)

        for output in outputs:
            print(output.size())

        outputs = torch.stack(outputs, dim=-2)
        print(outputs.size())
        rope_logits = 0
        qw, kw = outputs[...,:self.biaffine_size // 2], outputs[...,self.biaffine_size // 2:] 
        print(qw.size())
        print(kw.size())
        pos_emb = self.sinusoidal_position_embedding(4, 32, self.biaffine_size // 2)
        print(pos_emb.size())
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        print(cos_pos.size())
        print(sin_pos.size())
        qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        print(qw2.size())
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        print(qw.size())
        print(kw.size())
        RoPE_logits= torch.einsum('bmhd,bnhd->bhmn', qw, kw).permute(0, 2, 3, 1).squeeze(-1)


        return (self.down_layer(logits).squeeze(-1) + rope_logits) / (200 ** 0.5)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings
    

    def RoPE(self, dense):
        rope_logits = dense

        return rope_logits


model = Biaffine()

model.forward(inputs=inputs)