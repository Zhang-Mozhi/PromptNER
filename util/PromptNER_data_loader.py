import torch
import torch.utils.data as data
import os
from transformers import BertTokenizer
import random
from .utils import get_entities
from copy import copy
import json
from torch.utils.data.distributed import DistributedSampler

O_CLASS = 0


def get_O_type(s, e, entity_spans):
    O_type = 0
    for span in entity_spans:
        entity_s, entity_e = span
        if s >= entity_s and e <= entity_e:
            O_type = 2  # The sub span of one of entity spans
            break
        if (s <= entity_e and e > entity_e) or (s < entity_s and e >= entity_s):
            O_type = 1  # overlap with one of entity spans
            break
    return O_type  # other


class FewDataSet(data.Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, filepath, opt, finetune=False):

        self.FewNERD = True if opt.dataset == 'fewnerd' else False

        if self.FewNERD:
            filepath = f'{filepath}_{opt.N}_{opt.K}.jsonl'

        if not os.path.exists(filepath):
            print("[ERROR] Data file {} does not exist!".format(filepath))
            assert (0)

        print(f'load data from {filepath}')

        self.N = opt.N
        self.K = opt.K
        self.L = opt.L
        self.opt = opt
        self.is_support = True
        self.data_ratio = opt.data_ratio
        self.distinct_tags = None

        self.max_o_num = opt.max_o_num
        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)

        self.finetune = finetune
        self.tag2nl = {"loc": "location", "per": "person", "misc": "miscellaneous", "org": "organization", 'gpe':'geographical social political entity', 'norp':"nationality religion", "work_of_art":"work of art", "fac": 'facility', 'creative-work': 'creative work'}
        if self.FewNERD:
            datas = open(filepath).readlines()
            datas = [json.loads(x.strip()) for x in datas]

            if 'train11' in filepath and (opt.N == 10 and opt.K == 5):
                self.datas = []
                for d in datas:
                    support = copy(d['support'])
                    query_num = len(d['query']['word'])
        
                    for idx in range((query_num + opt.eposide_tasks - 1) // opt.eposide_tasks):
                        query = {}
                        for k in d['query']:
                            query[k] = d['query'][k][idx * opt.eposide_tasks:(idx + 1) * opt.eposide_tasks]
                        data_unit = {'support': support,
                                     'query': query,
                                     'types': d['types']}
                      
                        self.datas.append(data_unit)

            else:
                """
                self.datas = []
                # 如果不看xxx-other 性能如何？ 留着补ablation
                for d in datas:
                    support = copy(d['support'])
                    query = copy(d['query'])
                    data_unit = {'support': support,
                                 'query': query,
                                 'types': d['types']}
                    #print(d['types'])
                    flag = 1
                    for type in d['types']:
                        type = type.split('-')
                        if type[-1] == 'other':
                            flag = 0
                            break
                    if flag == 0:
                        flag = 1
                        continue
                    self.datas.append(data_unit)
                """
                self.datas = datas
        else: # snips
            datas = json.load(open(filepath))
            keys = list(datas.keys())
            self.datas = []
            for key in keys:
                print('[Field] {}'.format(key))
                if 'train' in filepath:
                    for d in datas[key]:
                        support = copy(d['support'])
                        
                        query_num = len(d['batch']['seq_ins'])
         
                        query = copy(d['batch'])
                    
                        data_unit = {'support': support, 'query': query}
                        self.datas.append(data_unit)
                        """
                        snips 数据量可能没有这么大 不用分eposide_tasks
                        assert query_num % opt.eposide_tasks == 0
                        for idx in range(query_num // opt.eposide_tasks):
                            query = {}
                            for k in d['batch']:
                                query[k] = d['batch'][k][idx * opt.eposide_tasks:(idx + 1) * opt.eposide_tasks]
                            data_unit = {'support': support,
                                         'query': query}
                    
                            self.datas.append(data_unit)
                        
                        """
                else:
                    self.datas += datas[key]
        data_num = int(len(self.datas) * self.data_ratio)
        self.datas = self.datas[0: data_num]
        self.train = True if 'train' in filepath else False
   
        print(f'eposide num: {len(self.datas)}')

    def __additem__(self, d, word, seq_len, gold_entitys, gold_spans, gold_span_tag,
                    gold_span_num, indices, tags, prefix_prompt_word, prefix_seq_len, prefix_indices):
        d['word'].append(word)
        d['seq_len'].append(seq_len)
        d['gold_entitys'].append(gold_entitys)
        d['gold_spans'].append(gold_spans)
        d['gold_span_tag'].append(gold_span_tag)
        d['gold_span_num'].append(gold_span_num)
        d['indices'].append(indices)
        d['tags'].append(tags)
        d['prefix_prompt_word'].append(prefix_prompt_word)
        d['prefix_seq_len'].append(prefix_seq_len)
        d['prefix_indices'].append(prefix_indices)

    def __fewnerd_get_class_span_dict__(self, label):
        '''
        返回一个sentence各个label对应的span 左闭右闭
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        from: https://github.com/thunlp/Few-NERD/blob/main/util/metric.py
        '''
        class_span = {}
        current_label = None
        i = 0
        # having tags in string format ['O', 'O', 'person-xxx', ..]
        while i < len(label):
            if label[i] != 'O':
                start = i
                end = i
                current_label = label[i]
                i += 1
                while i < len(label) and label[i] == current_label:
                    end = i
                    i += 1
                if current_label in class_span:
                    class_span[current_label].append((start, end))
                else:
                    class_span[current_label] = [(start, end)]
            else:
                i += 1
        return class_span

    def __snips_get_tags__(self, seq_labels):
        all_label = set()
        for seq_label in seq_labels:
            for l in seq_label:
                if l != 'O':
                    l = l[2:]
                    all_label.add(l)
        all_label = list(all_label)
        all_label.sort()
        return all_label

    def __get_gold_spans__(self, tags):
        """
        Returns:
            gold_entitys:  {[b,e,1], [b,e,3], .... }  a set
            gold_spans: [[b,e], [b,e], .... ]  gold_span_num x 2
            span_tags: [1, 3, 2, ...] gold_span_num
        """
        gold_spans = []
        gold_span_tag = []
       
        # Get all entity spans.
        gold_entitys = set()
        if self.FewNERD:
            class2spans = self.__fewnerd_get_class_span_dict__(tags)  # 左闭右闭 {label:[(start_pos, end_pos), ...]}
        else:
            class2spans = get_entities(tags)
        for cls in class2spans:
            spans_with_tag = class2spans[cls]
            for span in spans_with_tag:  # [(start_pos, end_pos), ...]
                b, e = span
                tag = self.tag2label[cls]
    
                assert tag >= self.opt.O_class_num
                if not self.is_support:
                    tag = tag - self.opt.O_class_num + 1
                gold_entitys.add((b, e, tag))  # 左闭右闭区间

                gold_spans.append([b, e])
                gold_span_tag.append(tag)

        return gold_entitys, gold_spans, gold_span_tag

    # 一个episode
    def __populate__(self, samples, isSupport=False, savelabeldic=False):
        """prefix_prompt_word: find some entities, such as non-entity, other, person, location, age: Alice May lives in Chicago."""
        dataset = {'word': [], 'seq_len': [],  'sentence_num': [],
                   'gold_entitys': [],  'gold_spans': [], 'gold_span_tag': [], 'gold_span_num': [],
                   'indices': [], 'tags': [], 'prefix_prompt_word': [], 'prefix_seq_len': [],
                   'prefix_indices': []}

        if self.FewNERD:
            words = [x for x in samples['word']]
            seq_tags = [x for x in samples['label']]
        else:
            words = [x for x in samples['seq_ins']]
            seq_tags = [x for x in samples['seq_outs']]

        # for query:
        #prefix_word = ['find', 'some', 'entities', ',' ,'such', 'as', 'non', '-', 'entity', ',']
        #query_prefix_indices = [0,0,0,0,0,0,0,1,1,1,0]
        if self.opt.prompt_id == 0:
            prefix_word = ['find', 'some', 'entities', ',' ,'such', 'as', 'none', ',']
            query_prefix_indices = [0,0,0,0,0,0,0,1,0]
        elif self.opt.prompt_id == 1:
            prefix_word = ['none', ',']
            query_prefix_indices = [0,1,0]
        elif self.opt.prompt_id == 2:
            prefix_word = ['your', 'task', 'is', 'to', 'extract', 'some', 'named', 'entities', ',', 'such', 'as','none', ',']
            # print("****"*20)
            # for p in prefix_word:
            #     prefix_piece = self.tokenizer.wordpiece_tokenizer.tokenize(p)
            #     print(p)
            #     print(prefix_piece)
            # exit(1)
            query_prefix_indices = [0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        else:
            prefix_word = ['find', 'some', 'entities', ',' ,'such', 'as', 'none', ',']
            query_prefix_indices = [0,0,0,0,0,0,0,1,0]
        num = len(self.distinct_tags)
        if not self.FewNERD:  # in snips dataset, the N will change 
            self.N  = num

        if self.FewNERD:
            for idx in range(num):
                tags = self.distinct_tags[idx]
                t = tags.replace('-', ' - ')
                t = t.replace('/', ' / ')
                p_list = t.split(" ")
                wp_len = 0
                for p in p_list:
                    prefix_piece = self.tokenizer.wordpiece_tokenizer.tokenize(p)
                    wp_len += len(prefix_piece)
                    prefix_word.extend(prefix_piece)
                query_prefix_indices.extend([idx+2] * wp_len)
                if idx == num-1:
                    prefix_word.extend(['[SEP]'])
                    query_prefix_indices.append(0)
                else:
                    prefix_word.extend(",")
                    query_prefix_indices.append(0)
        else: # snips / cross_ner
            for idx in range(num):
                tags = self.distinct_tags[idx].lower()
                if self.tag2nl.__contains__(tags):
                    tags = self.tag2nl[tags]
                #tags = self.tag2nl[tags]
                t = tags.replace('-', ' - ')
                t = t.replace('_', ' _ ')
                t = t.replace('/', ' / ')
                p_list = t.split(" ")
                wp_len = 0
                for p in p_list:
                    prefix_piece = self.tokenizer.wordpiece_tokenizer.tokenize(p)
                    wp_len += len(prefix_piece)
                    prefix_word.extend(prefix_piece)
                query_prefix_indices.extend([idx+2] * wp_len)
                if idx == num-1:
                    prefix_word.extend(['[SEP]'])
                    query_prefix_indices.append(0)
                else:
                    prefix_word.extend(",")
                    query_prefix_indices.append(0)
        prefix_len = len(prefix_word)

        for word, seq_tag in zip(words, seq_tags):
            word_expand = []
            seq_tag_expand = []

            indices = [0]
 
            prefix_indices = query_prefix_indices[:]
            prefix_prompt_word = prefix_word[:]

            for idx, (w, st) in enumerate(zip(word, seq_tag)):

                word_piece = self.tokenizer.wordpiece_tokenizer.tokenize(w)

                word_expand.extend(word_piece)
                prefix_prompt_word.extend(word_piece)
                seq_tag_expand.append(st)
                indices.append(idx+1)
  
                prefix_indices.append(idx+self.N+2)
                if len(word_piece) > 1:
                    for _ in word_piece[1:-1]:
                        indices.append(idx+1)
                        prefix_indices.append(idx+self.N+2)
                        seq_tag_expand.append(st.replace('B-', 'I-'))  # JUST For SNIPS，There does not exist `B-` in FewNERD
                    indices.append(idx+1)
                    prefix_indices.append(idx+self.N+2)
                    seq_tag_expand.append(st.replace('B-', 'I-'))  # JUST For SNIPS，There does not exist `B-` in FewNERD

         
            assert len(word_expand) == len(seq_tag_expand)
            if len(word_expand) > 110: # bert-base 只支持512个token，因为indices提前加上了cls和sep的indices，所以截断长度不一样
                word_expand = word_expand[:510]
                indices = indices[:511]
            if len(prefix_prompt_word) > 510:
                prefix_prompt_word = prefix_prompt_word[:510]
                prefix_indices = prefix_indices[:511]
            indices.append(0)
            prefix_indices.append(0)
            word = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + word_expand + ['[SEP]'])
            
            prefix_prompt_word = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + prefix_prompt_word + ['[SEP]'])
  
            seq_len = len(word)
            prefix_seq_len = len(prefix_prompt_word)

            gold_entitys, gold_spans, gold_span_tag = self.__get_gold_spans__(seq_tag)

            word = torch.tensor(word)#.long()
            prefix_prompt_word = torch.tensor(prefix_prompt_word)#.long()
            gold_span_num = len(gold_spans)

            indices = torch.tensor(indices, dtype=torch.long)  # torch_scatter need int64 for index tensor
            prefix_indices = torch.tensor(prefix_indices, dtype=torch.long) # torch_scatter need int64 for index tensor
            tags = list()

            self.__additem__(dataset, word, seq_len, gold_entitys, gold_spans, gold_span_tag,
                             gold_span_num, indices, tags, prefix_prompt_word, prefix_seq_len, prefix_indices)

        dataset['sentence_num'] = [len(dataset['word'])]

        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, index):
        index = index % len(self.datas)
        support = self.datas[index]['support']
        if 'query' in self.datas[index].keys():
            query = self.datas[index]['query']
        else:
            query = self.datas[index]['batch']
        if self.FewNERD:
            distinct_tags = ['O'] + self.datas[index]['types']
        else:
            self.datas[index]['types'] = self.__snips_get_tags__(support['seq_outs'])
            distinct_tags = ['O'] + self.datas[index]['types']
     
        self.distinct_tags = self.datas[index]['types']
        # label2tag 是按照distinct_tags的顺序来得到的
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        
        # print(self.tag2label)
        self.is_support = True
        support_set = self.__populate__(support, isSupport=self.is_support)
        self.is_support = False
        query_set = self.__populate__(query, isSupport=self.is_support, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return len(self.datas)


def collate_fn(data):
    batch_support = {'word': [], 'seq_len': [], 'sentence_num': [],
                     'gold_entitys': [],  'gold_spans': [], 'gold_span_tag': [], 'gold_span_num': [], 'indices': [],
                     'tags': [], 'prefix_prompt_word': [], 'prefix_seq_len': [], 'prefix_indices': []}
    batch_query = {'word': [], 'seq_len': [], 'sentence_num': [],
                   'gold_entitys': [],  'gold_spans': [], 'gold_span_tag': [], 'gold_span_num': [], 'indices': [],
                   'tags': [], 'prefix_prompt_word': [], 'prefix_seq_len': [], 'prefix_indices': []}

    support_sets, query_sets = zip(*data)

    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            batch_query[k] += query_sets[i][k]

    gold_tag_list = []
    for tags in batch_support['gold_span_tag']:
        for tag in tags:
            gold_tag_list.append(tag)

    batch_support['gold_span_tag'] = torch.tensor(gold_tag_list).long()
    batch_support['word'] = torch.nn.utils.rnn.pad_sequence(batch_support['word'], batch_first=True, padding_value=0)

    batch_support['indices'] = torch.nn.utils.rnn.pad_sequence(batch_support['indices'], batch_first=True,
                                                               padding_value=0)
    batch_support['prefix_indices'] = torch.nn.utils.rnn.pad_sequence(batch_support['prefix_indices'], batch_first=True,
                                                               padding_value=0)

    batch_support['prefix_prompt_word'] = torch.nn.utils.rnn.pad_sequence(batch_support['prefix_prompt_word'],
                                                                          batch_first=True, padding_value=0)

    gold_tag_list = []
    for tags in batch_query['gold_span_tag']:
        for tag in tags:
            gold_tag_list.append(tag)

    batch_query['gold_span_tag'] = torch.tensor(gold_tag_list).long()

    batch_query['word'] = torch.nn.utils.rnn.pad_sequence(batch_query['word'], batch_first=True, padding_value=0)

    batch_query['indices'] = torch.nn.utils.rnn.pad_sequence(batch_query['indices'], batch_first=True,
                                                             padding_value=0)
    batch_query['prefix_indices'] = torch.nn.utils.rnn.pad_sequence(batch_query['prefix_indices'], batch_first=True,
                                                                    padding_value=0)
    batch_query['prefix_prompt_word'] = torch.nn.utils.rnn.pad_sequence(batch_query['prefix_prompt_word'],
                                                                        batch_first=True, padding_value=0)
    return batch_support, batch_query


def get_loader(filepath, opt, num_workers=0, shuffle=True, finetune=False, is_distributed=None):

    dataset = FewDataSet(filepath, opt, finetune=finetune)

  
    #sampler = DistributedSampler(dataset) if is_distributed else None

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  sampler=None)
    return data_loader
