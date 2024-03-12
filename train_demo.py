import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import torch
import random
from thop import profile
from util.PromptNER_data_loader import get_loader
from util.framework import FewShotNERFramework
from model.NERModel import PromptNER


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inter',
                        help='training mode, must be in [inter, intra, supervised, 1, 2, 3, 4, ...]')
    parser.add_argument('--dataset', default='fewnerd',
                        help='training datasets, must be in [fewnerd, snips, ner]')

    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--model', default='PromptNER', type=str)

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--bert_lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='the rate between CE/CL loss')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')

    parser.add_argument('--max_epoch', default=30, type=int,
                        help='max_epoch')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimizer')

    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle train data')

    parser.add_argument('--hidsize', default=100, type=int,
                        help='dimension of hidden_size')

    parser.add_argument('--bert_path', default='bert-base-uncased', type=str,
                        help='bert-path')

    parser.add_argument('--max_o_num', default=500, type=int,
                        help='down-sampling for fewnerd for type O in each sentence')

    parser.add_argument('--num_heads', default=1, type=int,
                        help='multi-head-attention head num')

    parser.add_argument('--L', default=8, type=int,
                        help='max span length')

    parser.add_argument('--early_stop', default=20, type=int,
                        help='early_stop')

    parser.add_argument('--val_step', default=10, type=int)

    parser.add_argument('--soft_nms_k', default=1e-5, type=float)

    parser.add_argument('--soft_nms_u', default=1e-5, type=float)

    parser.add_argument('--soft_nms_delta', default=0.1, type=float)

    parser.add_argument('--beam_size', default=5, type=int)

    parser.add_argument('--warmup_step', default=300, type=int)

    parser.add_argument('--eposide_tasks', default=15, type=int)

    parser.add_argument('--data_ratio', default=1.0, type=float)

    parser.add_argument('--checkpoint_id', default=0, type=int)

    parser.add_argument('--threshold', default=0.5, type=float)

    parser.add_argument('--span_sample_ratio', default=0.2, type=float)

    parser.add_argument('--finetuning_steps', default=8, type=int)
    
    parser.add_argument('--kNN_ratio', default=2.0, type=float)
    
    parser.add_argument('--just_predict', default=0, type=int)

    parser.add_argument('--prompt_id', default=0, type=int)

    opt = parser.parse_args()

    N = opt.N
    K = opt.K
    model_name = opt.model
    opt.O_class_num = 1
    print(opt)
    print(opt.dataset)
    if opt.dataset == 'fewnerd':
        print("{}-way-{}-shot Few-Shot NER".format(N, K))
    elif opt.dataset == 'snips':
        print("{}-shot SNIPS".format(K))
    elif opt.dataset == 'cross-dataset':
        print("{}-shot Cross Datasets".format(K))
    else:
        print("Not support this dataset.")
        raise NotImplementedError

    print("model: {}".format(model_name))
    print('mode: {}'.format(opt.mode))

    set_seed(opt.seed)

    print('loading model and tokenizer...')

    print('loading data...')

    if opt.dataset == 'fewnerd':
        opt.train = f'data/episode-data/{opt.mode}/train'
        opt.val = f'data/episode-data/{opt.mode}/dev'
        opt.test = f'data/episode-data/{opt.mode}/test'
    elif opt.dataset == 'snips':
        if K == 5:
            root_dataset = 'data/xval_' + opt.dataset + '_shot_5'
            opt.train = f'{root_dataset}/{opt.dataset}-train-{opt.mode}-shot-5.json'
            opt.val = f'{root_dataset}/{opt.dataset}-valid-{opt.mode}-shot-5.json'
            opt.test = f'{root_dataset}/{opt.dataset}-test-{opt.mode}-shot-5.json'
        else:
            root_dataset = 'data/xval_' + opt.dataset
            opt.train = f'{root_dataset}/{opt.dataset}_train_{opt.mode}.json'
            opt.val = f'{root_dataset}/{opt.dataset}_valid_{opt.mode}.json'
            opt.test = f'{root_dataset}/{opt.dataset}_test_{opt.mode}.json'
    elif opt.dataset == 'cross-dataset':
        if K == 5:
            root_dataset = 'data/xval_ner_shot_5'
            opt.train = f'{root_dataset}/ner-train-{opt.mode}-shot-5.json'
            opt.val = f'{root_dataset}/ner-valid-{opt.mode}-shot-5.json'
            opt.test = f'{root_dataset}/ner-test-{opt.mode}-shot-5.json'
        else:
            root_dataset = 'data/xval_ner'
            opt.train = f'{root_dataset}/ner_train_{opt.mode}.json'
            opt.val = f'{root_dataset}/ner_valid_{opt.mode}.json'
            opt.test = f'{root_dataset}/ner_test_{opt.mode}.json'

    if opt.dataset == 'fewnerd':
        model = PromptNER(model_name=opt.bert_path, num_ner_tag=opt.N, K=opt.K, N=opt.N, mode=opt.mode, isCrossNER=False, kNN_ratio=opt.kNN_ratio)
    elif opt.dataset == 'cross-dataset':
        model = PromptNER(model_name=opt.bert_path, num_ner_tag=opt.N, K=opt.K, N=opt.N, mode=opt.mode, isCrossNER=True, kNN_ratio=opt.kNN_ratio)
    else:
        model = PromptNER(model_name=opt.bert_path, num_ner_tag=opt.N, K=opt.K, N=opt.N, mode=opt.mode, isCrossNER=True, kNN_ratio=opt.kNN_ratio)


    total = sum([param.nelement() for param in model.parameters()])
    print("Number of paramter: %.2fM" % (total / 1e6))

    if torch.cuda.is_available():
        model.cuda()

    prefix = opt.dataset + '-' + model.model_name

    if opt.dataset == 'fewnerd':
        prefix += f'-N_{opt.N}-K_{opt.K}-mode_{opt.mode}-drop_{opt.dropout}-lr_{opt.lr}-bertlr_{opt.bert_lr}-hidsize_{opt.hidsize}-graditer_{opt.grad_iter}-es_{opt.early_stop}-warmup_{opt.warmup_step}-eptasks_{opt.eposide_tasks}'
    elif opt.dataset == 'fewnerdSUP':
        prefix += f'-Dataset_{opt.dataset}-drop_{opt.dropout}-lr_{opt.lr}-bertlr_{opt.bert_lr}-hidsize_{opt.hidsize}-graditer_{opt.grad_iter}-es_{opt.early_stop}-warmup_{opt.warmup_step}-eptasks_{opt.eposide_tasks}'
    else:
        prefix += f'-K_{opt.K}-mode_{opt.mode}-drop_{opt.dropout}-lr_{opt.lr}-bertlr_{opt.bert_lr}-hidsize_{opt.hidsize}-graditer_{opt.grad_iter}-es_{opt.early_stop}-warmup_{opt.warmup_step}-eptasks_{opt.eposide_tasks}'
    if opt.shuffle:
        prefix += '-sff'
    prefix += '-maxonum_{}'.format(opt.max_o_num)
    prefix += '-seed_{}'.format(opt.seed)
    ckpt = 'checkpoint/ID:{}_{}.pth.tar'.format(opt.checkpoint_id, prefix)
    first_stage_ckpt = 'checkpoint/first_stage_ID:{}_{}.pth.tar'.format(opt.checkpoint_id, prefix)
        
    opt.batch_size = 1

    train_data_loader = get_loader(opt.train, opt, shuffle=True, is_distributed=True)

    val_data_loader = get_loader(opt.val, opt, shuffle=False, is_distributed=False)
    test_data_loader = get_loader(opt.test, opt, shuffle=False, is_distributed=False)
    if opt.mode == "inter":
        predict_threshold = 0.25
    else:
        predict_threshold = 0.25
    framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, opt, predict_threshold)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    print("*" * 20)
    print(opt)
    print("*" * 20)
    print('*' * 20)
    print('[save_ckpt]: {}'.format(ckpt))
    print('[first_stage_save_ckpt]: {}'.format(first_stage_ckpt))
    print('*' * 20)

    if "inter" in ckpt:
        model.predict_threshold = 0.1  
    elif "intra" in ckpt:
        model.predict_threshold = 0.05 
    else:
        model.predict_threshold = 0.15
    
    if opt.just_predict == 0:
        if opt.K == 5 and (opt.mode == 'inter' or opt.mode == 'intra'):
            framework.train_two_stage(model=model,
                            model_name=prefix,
                            opt=opt,
                            save_ckpt=ckpt,
                            first_stage_ckpt=first_stage_ckpt,
                            warmup_step=500,
                            train_step=5000,
                            just_train_second_stage=False,
                            just_train_first_stage=False)  # 分开train span_detector和classifier
        elif opt.mode == 'inter' or opt.mode == 'intra':
            framework.train(model=model,
                        model_name=prefix,
                        opt=opt,
                        save_ckpt=ckpt,
                        warmup_step=1500,
                        train_step=15000)# 一起train span_detector和classifier
        elif opt.dataset == 'snips':
            if opt.K == 1:
                framework.train(model=model,
                                model_name=prefix,
                                opt=opt,
                                save_ckpt=ckpt,
                                warmup_step=500,
                                train_step=5000)# 一起train span_detector和classifier
            else: # k == 5
                framework.train(model=model,
                                model_name=prefix,
                                opt=opt,
                                save_ckpt=ckpt,
                                warmup_step=500,
                                train_step=5000)# 一起train span_detector和classifier
        else:
            if opt.K == 1:
                framework.train(model=model,
                                model_name=prefix,
                                opt=opt,
                                save_ckpt=ckpt,
                                warmup_step=400,
                                train_step=4000)# 一起train span_detector和classifier
            else: # k == 5
                framework.train(model=model,
                                model_name=prefix,
                                opt=opt,
                                save_ckpt=ckpt,
                                warmup_step=200,
                                train_step=2000)# 一起train span_detector和classifier
    

    if "inter" in ckpt:
        model.predict_threshold = 0.01 #  0.01  recall more candidate spans
    elif "intra" in ckpt:
        model.predict_threshold = 0.01 #  0.01
    else:
        model.predict_threshold = 0.01 #  0.01

    res = framework.test(model, opt=opt, ckpt=ckpt, L=opt.L, use_sgd_for_bert=False, finetuning_steps=opt.finetuning_steps)
    if not os.path.exists('./results'):
        os.mkdir('results')
    if opt.dataset == 'snips':
        result_path = 'results/{}_{}_K{}_result.txt'.format(opt.dataset, opt.mode, opt.K)
    else:
        result_path = 'results/{}_{}_N{}_K{}_DataRatio:{}_result.txt'.format(opt.dataset, opt.mode, opt.N, opt.K, opt.data_ratio)
    with open(result_path, 'a') as f:
        f.write(prefix + f'-kNN_ratio_{opt.kNN_ratio}-Prompt_id_{opt.prompt_id}' + '\n')
        f.write("{}\n".format(res))
    print(ckpt)

if __name__ == "__main__":
    main()

