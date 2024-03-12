import os
import numpy as np
import sys
import time
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

#from transformers import AdamW
from transformers import  get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LinearLR
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import time
from .utils import now, get_p_r_f1, metrics_by_entity_tuples, get_fp, metrics_by_span_detector


O_CLASS = 0


class FewShotNERFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, opt, predict_threshold):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        #self.train_data_loader_for_span = train_data_loader_for_span
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.opt = opt
        self.predict_threshold = predict_threshold

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def overlap(self, left, right, span_list):
        for (l, r) in span_list:
            if right < l or r < left:
                continue
            else:
                return True
        return False

    def get_entity_tuples_from_span(self, score_list, pred_list, span_list, threshold=0.5):
        """
        Args:
            score_list: batch_size x span_num_of_each_sent (tensor lists without padding)
            pred_list: batch_size x span_num_of_each_sent (tensor lists without padding)
            span_list: batch_size x span_num_of_each_sent x 2 (begin, end) (tensor lists without padding)
        Returns:
            最小元素为一个元组 （b, e, tag） b, e 表示entity的左闭右闭区间
            episode_entity_tuples :[query_sent_num, entity_nums] (list->tuples)
        """

        # 每个batch

        episode_entity_tuples = []
        if score_list is None:
            return episode_entity_tuples

        for sent_idx in range(len(score_list)):

            one_sent_tuples = set()
            score = score_list[sent_idx]
            pred = pred_list[sent_idx]

            span = span_list[sent_idx]
       
            if len(span) != 0:
                sent_list = list((s, p, (b, e)) for s, p, [b, e] in zip(score, pred, span))
                sent = sorted(sent_list, key=lambda tup: tup[0], reverse=True)
                #print("_________________")
                #torch.set_printoptions(threshold=np.inf)
                #print(sent)
                cur_span_list = []
                cur_threshold = threshold
                for idx, (score, label, (span_left, span_right)) in enumerate(sent):
                    #if score <= threshold:
                    if score < cur_threshold:      
                        continue
                    if not self.overlap(span_left, span_right, cur_span_list):
                        if label != O_CLASS:
                            cur_span_list.append((span_left, span_right))
                            one_sent_tuples.add((span_left, span_right, label))
                    
                while len(one_sent_tuples) < 1 and cur_threshold > 0.1:
                    cur_threshold -= 0.01
                    for idx, (score, label, (span_left, span_right)) in enumerate(sent):
                        #if score <= threshold:
                        if score < cur_threshold:      
                            continue
                        if not self.overlap(span_left, span_right, cur_span_list):
                            if label != O_CLASS:
                                cur_span_list.append((span_left, span_right))
                                one_sent_tuples.add((span_left, span_right, label))
                                break
                            
            episode_entity_tuples.append(one_sent_tuples)
            #print(one_sent_tuples)
            #print("_________________")

        return episode_entity_tuples

    def train_two_stage(self,
              model,
              model_name,
              opt,
              save_ckpt=None,
              first_stage_ckpt=None,
              warmup_step=500,
              train_step=5000,
              just_train_second_stage=False,
              just_train_first_stage=False
              ):

        ln_params = []
        non_ln_params = []
        non_pretrain_params = []
        non_pretrain_ln_params = []

        for name, param in model.named_parameters():
            name = name.lower()
            if param.requires_grad is False:
                continue
        
            if 'proto_model' in name: 
                param.requires_grad = False
            elif  'pretrain_model' in name:
                if 'norm' in name or 'bias' in name:
                    ln_params.append(param)
                else:
                    non_ln_params.append(param)
            else:
                if 'norm' in name or 'bias' in name:
                    non_pretrain_ln_params.append(param)
                else:
                    non_pretrain_params.append(param)

        weight_decay = 1e-2
        
        parameters_to_optimize = [{'params': non_ln_params, 'lr': opt.bert_lr, 'weight_decay': weight_decay},
                                  {'params': ln_params, 'lr': opt.bert_lr, 'weight_decay': 0},
                                  {'params': non_pretrain_ln_params, 'lr': opt.lr, 'weight_decay': 0},
                                  {'params': non_pretrain_params, 'lr': opt.lr, 'weight_decay': weight_decay}]

        optimizer = AdamW(params=parameters_to_optimize, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_step)
        scaler = GradScaler()

        # load model
        if just_train_second_stage:
            state_dict = self.__load_model__(first_stage_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)
        else:
            ## --------- first stage --------- 
            model.train()
            # Training
            best_f1 = 0
            patient = 0
            it = 1
            epoch = 1
            step = 0
            grad_iter = 1

            if self.opt.dataset == 'fewnerd':
                while step < train_step:
                    epoch_loss = 0
                    episode_num = 0
                    for _, (support, query) in tqdm(enumerate(self.train_data_loader)):
                        
                        optimizer.zero_grad()

                        torch.cuda.empty_cache()
                    
                        support['word'] = support['word'].cuda()
                        support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                        support['indices'] = support['indices'].cuda()
                        support['prefix_indices'] = support['prefix_indices'].cuda()

                        query['word'] = query['word'].cuda()
                        query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                        query['indices'] = query['indices'].cuda()
                        query['prefix_indices'] = query['prefix_indices'].cuda()
                        with autocast():
                            res = model.train_step_stage_one(support, query)
                            loss = res['loss'] 

                        epoch_loss += loss.item()
                        episode_num += 1
                        step += 1
                        
                        sys.stdout.write(f" [Stage 1]: epoch: {epoch} | step: {it} | episode loss: {loss :.5f} | epoch loss:{epoch_loss / episode_num: .5f} | Type_loss:{res['Type_loss'] :.5f}, Span_loss:{res['Span_loss'] :.5f} CL_loss:{(res['support_CL_loss'] + res['query_CL_loss']) / 2 :.5f}  \r")
                        sys.stdout.flush()

                        #loss.backward()
                        scaler.scale(loss).backward()
                        if it % grad_iter == 0:

                            """
                            optimizer.step()
                            optimizer.zero_grad()
                            scheduler.step()
                            """     
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                          
                    
                        if it  % (opt.val_step) == 0:
                            _, _, f1,  _, _, span_f1 = self.eval(model, L=opt.L)
                            torch.cuda.empty_cache()
                            #fitlog.add_metric({"dev": {"F1": f1}}, step=step)
                            model.train()
                            patient += 1
                            if span_f1 > best_f1:
                                print('Best checkpoint')
                                torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, first_stage_ckpt)
                                best_f1 = span_f1
                                #fitlog.add_best_metric({"dev": {"F1": best_f1}})

                                patient = 0
                            print('[Patient] {} / {}'.format(patient, opt.early_stop))
                            if patient >= opt.early_stop:
                                break
                        if it > train_step:
                            break
                                
                        it += 1
                    if patient >= opt.early_stop or it > train_step:
                        break
                    epoch += 1
                    
            else:
                while step < train_step:
                    epoch_loss = 0
                    episode_num = 0
                    # training module for snips
                    print(f'\nEpoch : {epoch}')
                    for _, (support, query) in tqdm(enumerate(self.train_data_loader)):

                        optimizer.zero_grad()

                        torch.cuda.empty_cache()
                        support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                        support['indices'] = support['indices'].cuda()
                        support['prefix_indices'] = support['prefix_indices'].cuda()

                        query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                        query['indices'] = query['indices'].cuda()
                        query['prefix_indices'] = query['prefix_indices'].cuda()

                        with autocast():
                            res = model.train_step_stage_one(support, query)
                            loss = res['loss'] / float(opt.grad_iter)

                        epoch_loss += loss
                        episode_num += 1
                        step += 1
                        it += 1
                        sys.stdout.write(f"epoch: {epoch} | step: {it} | episode loss: {loss :.5f} | epoch loss:{epoch_loss / episode_num: .5f} | Type_loss:{res['Type_loss'] :.5f}, Span_loss:{res['Span_loss'] :.5f} CL_loss:{(res['support_CL_loss'] + res['query_CL_loss']) / 2 :.5f}  \r")
                        sys.stdout.flush()

                        scaler.scale(loss).backward()
                        if it % opt.grad_iter == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
    
                        if it % opt.val_step == 0:
                            torch.cuda.empty_cache()
                            _, _, f1, _, _, span_f1 = self.eval(model, L=opt.L)

                            model.train()
                            patient += 1
                            if span_f1 > best_f1:
                                print('Best checkpoint')
                                torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, first_stage_ckpt)
                                best_f1 = span_f1
                                patient = 0
                            print('[Patient] {} / {}'.format(patient, opt.early_stop))
                            if patient >= opt.early_stop:
                                break

                    epoch += 1
                    it += 1

                    if patient >= opt.early_stop or epoch >= self.opt.max_epoch or step >= train_step:
                        break

            print("\n#################################\n")
            print(" Finish First Stage Training ")
            print("\n#################################\n")

        if just_train_first_stage:
            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, save_ckpt)
            return None
        ## --------- second stage --------- 
        if not just_train_second_stage:
            state_dict = self.__load_model__(first_stage_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)

        ln_params = []
        non_ln_params = []
        non_pretrain_params = []
        non_pretrain_ln_params = []
        for name, param in model.named_parameters():
            name = name.lower()
            if 'proto_model' in name: 
                param.requires_grad = True
                if 'norm' in name or 'bias' in name:
                    ln_params.append(param)
                else:
                    non_ln_params.append(param) 
            else:
                param.requires_grad = False

        weight_decay = 1e-2
        
        parameters_to_optimize = [{'params': non_ln_params, 'lr': opt.bert_lr, 'weight_decay': weight_decay},
                                  {'params': ln_params, 'lr': opt.bert_lr, 'weight_decay': 0},
                                  {'params': non_pretrain_ln_params, 'lr': opt.lr, 'weight_decay': 0},
                                  {'params': non_pretrain_params, 'lr': opt.lr, 'weight_decay': weight_decay}]
            

        optimizer = AdamW(params=parameters_to_optimize, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_step)
        scaler = GradScaler()
        
        model.train()
        # Training
        best_f1 = 0
        patient = 0
        it = 1
        epoch = 1
        step = 0
        grad_iter = 1

        if self.opt.dataset == 'fewnerd':
            while step < train_step:
                epoch_loss = 0
                episode_num = 0
                for _, (support, query) in tqdm(enumerate(self.train_data_loader)):
                    
                    optimizer.zero_grad()

                    torch.cuda.empty_cache()
                
                    support['word'] = support['word'].cuda()
                    support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                    support['indices'] = support['indices'].cuda()
                    support['prefix_indices'] = support['prefix_indices'].cuda()

                    query['word'] = query['word'].cuda()
                    query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                    query['indices'] = query['indices'].cuda()
                    query['prefix_indices'] = query['prefix_indices'].cuda()
                    with autocast():
                        res = model.train_step_stage_two(support, query)
                        loss = res['loss']

                    epoch_loss += loss.item()
                    episode_num += 1
                    step += 1
                      
                    sys.stdout.write(f" [Stage2] epoch: {epoch} | step: {it} | episode loss: {loss :.5f} | epoch loss:{epoch_loss / episode_num: .5f} | Type_loss:{res['Type_loss'] :.5f}, Span_loss:{res['Span_loss'] :.5f} CL_loss:{(res['support_CL_loss'] + res['query_CL_loss']) / 2 :.5f}  \r")
                    sys.stdout.flush()

                    #loss.backward()
                    scaler.scale(loss).backward()
                    if it % grad_iter == 0:

                        """
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        """     
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
               
                    if it % opt.val_step == 0:
                        _, _, f1, _, _, span_f1 = self.eval(model, L=opt.L)
                        torch.cuda.empty_cache()
                        #fitlog.add_metric({"dev": {"F1": f1}}, step=step)
                        model.train()
                        patient += 1
                        if f1 > best_f1:
                            print('Best checkpoint')
                            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, save_ckpt)
                            best_f1 = f1
                            #fitlog.add_best_metric({"dev": {"F1": best_f1}})

                            patient = 0
                        print('[Patient] {} / {}'.format(patient, opt.early_stop))
                    if patient >= opt.early_stop or it > train_step:
                        break

                    it += 1
                if patient >= opt.early_stop or it > train_step:
                    break
                epoch += 1
                
        else:
            while step < train_step:
                epoch_loss = 0
                episode_num = 0
                # training module for snips
                print(f'\nEpoch : {epoch}')
                for _, (support, query) in tqdm(enumerate(self.train_data_loader)):

                    optimizer.zero_grad()

                    torch.cuda.empty_cache()
                    support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                    support['indices'] = support['indices'].cuda()
                    support['prefix_indices'] = support['prefix_indices'].cuda()

                    query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                    query['indices'] = query['indices'].cuda()
                    query['prefix_indices'] = query['prefix_indices'].cuda()

                    with autocast():
                        res = model.train_step_stage_two(support, query)
                        loss = res['loss'] / float(opt.grad_iter)

                    epoch_loss += loss
                    episode_num += 1
                    step += 1
                    it += 1
                    sys.stdout.write(f"epoch: {epoch} | step: {it} | episode loss: {loss :.5f} | epoch loss:{epoch_loss / episode_num: .5f} | Type_loss:{res['Type_loss'] :.5f}, Span_loss:{res['Span_loss'] :.5f} CL_loss:{(res['support_CL_loss'] + res['query_CL_loss']) / 2 :.5f}  \r")
                    sys.stdout.flush()

                    scaler.scale(loss).backward()
                    if it % opt.grad_iter == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
 
                    if it % opt.val_step == 0:
                        torch.cuda.empty_cache()
                        _, _, f1, _, _, span_f1 = self.eval(model, L=opt.L)

                        model.train()
                        patient += 1
                        if f1 > best_f1:
                            print('Best checkpoint')
                            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, save_ckpt)
                            best_f1 = f1
                            patient = 0
                        print('[Patient] {} / {}'.format(patient, opt.early_stop))
                        if patient >= opt.early_stop:
                            break

                epoch += 1
        
                if patient >= opt.early_stop or epoch >= self.opt.max_epoch or it > train_step:
                    break

        print("\n#################################\n")
        print(" Finish training " + model_name)

    def train(self,
              model,
              model_name,
              opt,
              save_ckpt=None,
              warmup_step=500,
              train_step=5000
              ):

        ln_params = []
        non_ln_params = []
        proto_ln_params = []
        proto_non_ln_params = []
        non_pretrain_params = []
        non_pretrain_ln_params = []

        for name, param in model.named_parameters():
            name = name.lower()
        
            if param.requires_grad is False:
                continue
            if 'pretrain_model' or 'proto_model' in name: # Two different encoders for span detection and metion classifier respectively.
                if 'norm' in name or 'bias' in name:
                    ln_params.append(param)
                else:
                    non_ln_params.append(param)

            else:
                if 'norm' in name or 'bias' in name:
                    non_pretrain_ln_params.append(param)
                else:
                    non_pretrain_params.append(param)
            """
            elif 'proto_model' in name:
                if 'norm' in name or 'bias' in name:
                    proto_ln_params.append(param)
                else:
                    proto_non_ln_params.append(param)
            """
        weight_decay = 1e-2

        parameters_to_optimize = [{'params': non_ln_params, 'lr': opt.bert_lr, 'weight_decay': weight_decay},
                                  {'params': ln_params, 'lr': opt.bert_lr, 'weight_decay': 0},
                                  #{'params': proto_non_ln_params, 'lr': opt.bert_lr, 'weight_decay': weight_decay},
                                  #{'params': proto_ln_params, 'lr': opt.bert_lr, 'weight_decay': 0},
                                  {'params': non_pretrain_ln_params, 'lr': opt.lr, 'weight_decay': 0},
                                  {'params': non_pretrain_params, 'lr': opt.lr, 'weight_decay': weight_decay}]
            
        
        optimizer = AdamW(params=parameters_to_optimize, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_step)
        scaler = GradScaler()

        # load model
        if opt.load_ckpt:
            state_dict = self.__load_model__(opt.load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, opt.load_ckpt))
                own_state[name].copy_(param)

        model.train()
        # Training
        best_f1 = 0.
        patient = 0
        it = 1
        epoch = 1
        step = 0
        grad_iter = 1

        if self.opt.dataset == 'fewnerd':
            while step < train_step:
                epoch_loss = 0
                episode_num = 0
                for _, (support, query) in tqdm(enumerate(self.train_data_loader)):
                    
                    optimizer.zero_grad()

                    torch.cuda.empty_cache()
                
                    #support['word'] = support['word'].cuda()
                    support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                    support['indices'] = support['indices'].cuda()
                    support['prefix_indices'] = support['prefix_indices'].cuda()

                    #query['word'] = query['word'].cuda()
                    query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                    query['indices'] = query['indices'].cuda()
                    query['prefix_indices'] = query['prefix_indices'].cuda()
                    with autocast():
                        res = model.train_step_batch_NN(support, query)
                        loss = res['loss'] / float(grad_iter)

                    epoch_loss += loss
                    episode_num += 1
                    step += 1
                      
                    sys.stdout.write(f"epoch: {epoch} | step: {it} | episode loss: {loss :.5f} | epoch loss:{epoch_loss / episode_num: .5f} | Type_loss:{res['Type_loss'] :.5f}, Span_loss:{res['Span_loss'] :.5f} CL_loss:{(res['support_CL_loss'] + res['query_CL_loss']) / 2 :.5f}  \r")
                    sys.stdout.flush()

                    #loss.backward()
                    scaler.scale(loss).backward()
                    if it % grad_iter == 0:

                        """
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        """     
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        
                    
                    if it % opt.val_step == 0:
                        torch.cuda.empty_cache()
                        _, _, f1, _, _, span_f1 = self.eval(model, L=opt.L)

                        #fitlog.add_metric({"dev": {"F1": f1}}, step=step)
                        model.train()
                        patient += 1
                        if f1 > best_f1:
                            print('Best checkpoint')
                            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, save_ckpt)
                            best_f1 = f1
                            #fitlog.add_best_metric({"dev": {"F1": best_f1}})

                            patient = 0
                        print('[Patient] {} / {}'.format(patient, opt.early_stop))
                        if patient >= opt.early_stop:
                            break

                    it += 1
                if patient >= opt.early_stop:
                    break
                epoch += 1
                
        else:
            while step < train_step:
                epoch_loss = 0
                episode_num = 0
                # training module for snips
                print(f'\nEpoch : {epoch}')
                for _, (support, query) in tqdm(enumerate(self.train_data_loader)):

                    optimizer.zero_grad()

                    torch.cuda.empty_cache()
                    support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                    support['indices'] = support['indices'].cuda()
                    support['prefix_indices'] = support['prefix_indices'].cuda()

                    query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                    query['indices'] = query['indices'].cuda()
                    query['prefix_indices'] = query['prefix_indices'].cuda()

                    with autocast():
                        res = model.train_step_batch_NN(support, query)
                        loss = res['loss'] / float(opt.grad_iter)

                    epoch_loss += loss
                    episode_num += 1
                    step += 1
                    it += 1
                    sys.stdout.write(f"epoch: {epoch} | step: {it} | episode loss: {loss :.5f} | epoch loss:{epoch_loss / episode_num: .5f} | Type_loss:{res['Type_loss'] :.5f}, Span_loss:{res['Span_loss'] :.5f} CL_loss:{(res['support_CL_loss'] + res['query_CL_loss']) / 2 :.5f}  \r")
                    sys.stdout.flush()

                    scaler.scale(loss).backward()
                    if it % opt.grad_iter == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
 
                    if it % opt.val_step == 0:
                        torch.cuda.empty_cache()
                        _, _, f1, _, _, span_f1 = self.eval(model, L=opt.L)

                        model.train()
                        patient += 1
                        if f1 > best_f1:
                            print('Best checkpoint')
                            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'f1': f1, 'train_step': it}, save_ckpt)
                            best_f1 = f1
                            patient = 0
                        print('[Patient] {} / {}'.format(patient, opt.early_stop))
                        if patient >= opt.early_stop:
                            break

                epoch += 1

                if patient >= opt.early_stop or epoch >= self.opt.max_epoch or it > train_step:
                    break


        print("\n#################################\n")
        print(" Finish training " + model_name)

    def eval(self,
             model,
             ckpt=None,
             L=8,
             mode='val',
            ):
        print("")
        model.eval()

        if ckpt is None:
            if mode == 'val':
                print("Use val dataset")
                eval_dataset = self.val_data_loader
            else:
                print("Use test dataset")
                eval_dataset = self.test_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                checkpoint = self.__load_model__(ckpt)
                state_dict = checkpoint['state_dict']
                dev_f1 = checkpoint['f1']
                train_step = checkpoint.get('train_step', 'xx')
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_sample = 0.0

        iter_precision = 0.0

        iter_recall = 0.0

        iter_f1 = 0.0

        all_correct_cnt = 0
        all_pred_cnt = 0
        all_gold_cnt = 0
        all_fp_span_cnt = 0
        all_fp_type_cnt = 0
        all_correct_span_cnt = 0
        all_pred_span_cnt = 0
        all_false_cnt = 0
        final_all_correct_span_cnt = 0
        final_all_pred_span_cnt = 0
        final_share_begin_cnt = 0
        final_share_end_end = 0

        all_long = 0
        all_short = 0
        all_wo_overlap = 0
        all_threshold_cnt = 0
        
 
        with torch.no_grad():
            for _, (support, query) in tqdm(enumerate(eval_dataset)):

                torch.cuda.empty_cache()
                #support['word'] = support['word'].cuda()
                support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
                support['indices'] = support['indices'].cuda()
                support['prefix_indices'] = support['prefix_indices'].cuda()

                #query['word'] = query['word'].cuda()
                query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
                query['indices'] = query['indices'].cuda()
                query['prefix_indices'] = query['prefix_indices'].cuda()

                res = model.evaluate_step_NN(support, query)
        
     
                score, span, pred = res['score'], res['span'], res['pred']
                gold_entitys = query['gold_entitys']

                pred_entitys_origin = self.get_entity_tuples_from_span(score, pred, span, self.predict_threshold)

                correct_cnt, pred_cnt, label_cnt, false_cnt, fp_span_cnt, fp_type_cnt, final_correct_span_cnt, \
                    final_pred_span_cnt = metrics_by_entity_tuples(gold_entitys, pred_entitys_origin)  # (span_l, span_r, pred), (span_l, span_r, pred)
                label_cnt, correct_span_cnt, pred_span_cnt, long_cnt, short_cnt, wo_overlap_cnt,\
                    share_begin_cnt, share_end_cnt = metrics_by_span_detector(gold_entitys, span)
                
                precision, recall, f1 = get_p_r_f1(correct_cnt, pred_cnt, label_cnt)

                iter_precision += precision
                iter_recall += recall
                iter_f1 += f1

                all_correct_cnt += correct_cnt
                all_pred_cnt += pred_cnt
                all_gold_cnt += label_cnt
                all_fp_span_cnt += fp_span_cnt
                all_fp_type_cnt += fp_type_cnt
                all_correct_span_cnt += correct_span_cnt
                all_pred_span_cnt += pred_span_cnt
                all_false_cnt += false_cnt

                final_all_correct_span_cnt += final_correct_span_cnt
                final_all_pred_span_cnt += final_pred_span_cnt
                final_share_begin_cnt += share_begin_cnt
                final_share_end_end += share_end_cnt

                all_long += long_cnt
                all_short += short_cnt
                all_wo_overlap += wo_overlap_cnt

                iter_sample += 1

            all_p, all_r, all_f1 = get_p_r_f1(all_correct_cnt, all_pred_cnt, all_gold_cnt)

            all_span_p, all_span_r, all_span_f1 = get_p_r_f1(all_correct_span_cnt, all_pred_span_cnt, all_gold_cnt)
            final_all_span_p, final_all_span_r, final_all_span_f1 = get_p_r_f1(final_all_correct_span_cnt, final_all_pred_span_cnt, all_gold_cnt)
            all_fp_span, all_fp_type = get_fp(all_false_cnt, all_fp_span_cnt, all_fp_type_cnt)
      
            res_string = '''{} || [EVAL] || [FewNERD] : ( P: {:.4f}; R: {:.4f}; F1: {:.4f} FalseTotal:{}, FalseRation:{:.4f}, FP_span: {:.4f} FP_type: {:.4f} \n \t P_span: {:.4f}; R_span: {:.4f}; F1_span: {:.4f}, Final_P_span: {:.4f}; Final_R_span: {:.4f}; Final_F1_span: {:.4f}; Longer_span: {}, Shorter_span:{}, WO_overlap:{}, Same begin token:{}, Same end token:{})'''.format(
                now(), all_p, all_r, all_f1, all_false_cnt, all_false_cnt / (all_pred_cnt + 1e-6), all_fp_span, all_fp_type,
                all_span_p, all_span_r, all_span_f1, final_all_span_p, final_all_span_r, final_all_span_f1, all_long,
                all_short, all_wo_overlap, final_share_begin_cnt, final_share_end_end)
            sys.stdout.write(res_string + '\r')

            sys.stdout.flush()
            print("")

        if ckpt is None:
            return all_p, all_r, all_f1, all_span_p, all_span_r, all_span_f1

        else:
            return "Dev f1: {}; train_step: {} || ".format(dev_f1, train_step) + res_string

    def test(self,
             model,
             opt,
             ckpt=None,
             L=8,
             mode='test',
             use_sgd_for_bert=True,
             finetuning_steps=8):


        model.eval()
        checkpoint = None

        if ckpt is None:
            if mode == 'val':
                print("Use val dataset")
                eval_dataset = self.val_data_loader
            else:
                print("Use test dataset")
                eval_dataset = self.test_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                checkpoint = self.__load_model__(ckpt)
                state_dict = checkpoint['state_dict']
                dev_f1 = checkpoint['f1']
                train_step = checkpoint.get('train_step', 'xx')
                own_state = model.state_dict()

                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_sample = 0.0
        iter_precision = 0.0
        iter_recall = 0.0
        iter_f1 = 0.0

        all_correct_cnt = 0
        all_pred_cnt = 0
        all_gold_cnt = 0
        all_fp_span_cnt = 0
        all_fp_type_cnt = 0
        all_correct_span_cnt = 0
        all_pred_span_cnt = 0
        all_false_cnt = 0

        for _, (support, query) in tqdm(enumerate(eval_dataset)):

            model.train()
            #support['word'] = support['word'].cuda()
            support['prefix_prompt_word'] = support['prefix_prompt_word'].cuda()
            support['indices'] = support['indices'].cuda()
            support['prefix_indices'] = support['prefix_indices'].cuda()

            #query['word'] = query['word'].cuda()
            query['prefix_prompt_word'] = query['prefix_prompt_word'].cuda()
            query['indices'] = query['indices'].cuda()
            query['prefix_indices'] = query['prefix_indices'].cuda()

            ln_params = []
            non_ln_params = []
            non_pretrain_params = []
            non_pretrain_ln_params = []

            for name, param in model.named_parameters():
                name = name.lower()
                if param.requires_grad is False:  # 如果之前用了两阶段训练，有一些model的参数被冻住了，这里需要解冻，否则inference的时候效果不好。
                   param.requires_grad = True
             
                if 'pretrain_model' or 'proto_model' in name: # Two different encoders for span detection and metion classifier respectively.
                    if 'norm' in name or 'bias' in name:
                        ln_params.append(param)
                    else:
                        non_ln_params.append(param)
                else:
                    if 'norm' in name or 'bias' in name:
                        non_pretrain_ln_params.append(param)
                    else:
                        non_pretrain_params.append(param)

            weight_decay = 1e-2
            opt.lr = 2e-3
            parameters_to_optimize = [{'params': non_ln_params, 'lr': opt.bert_lr , 'weight_decay': weight_decay},
                                      {'params': ln_params, 'lr': opt.bert_lr, 'weight_decay': 0},
                                      {'params': non_pretrain_ln_params, 'lr': opt.lr, 'weight_decay': 0},
                                      {'params': non_pretrain_params, 'lr': opt.lr, 'weight_decay': weight_decay}]
        
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=opt.lr)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
         
            episode_loss = 1e7
            step = 0
          
            if opt.N == 10 and opt.K == 5:
                while True:
                    torch.cuda.empty_cache()
            
                    cur_loss = model.fine_tuning_step_stage_one(support)['loss']
    
                    if step >= finetuning_steps//2 or episode_loss <= 5e-2:
                        break
                    else:
                        cur_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                        episode_loss = cur_loss.item()
                        step += 1
                episode_loss = 1e7      
                step = 0
    
                while True:
                    torch.cuda.empty_cache()
                    cur_loss = model.fine_tuning_step_stage_two(support)['loss']
                    if step >= finetuning_steps//2 or episode_loss <= 5e-2:
                        break
                    else:
          
                        cur_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        episode_loss = cur_loss.item()
                        step += 1
            else:
                while True:
                    torch.cuda.empty_cache()

                    cur_loss = model.fine_tuning_step(support)['loss']
                    #print(cur_loss)
                    
                    if step >= finetuning_steps or episode_loss <= 1e-2:
                    #if step >= finetuning_steps:
                        break
                    else:
                        episode_loss = cur_loss.item()
                        cur_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        step += 1

            model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                res = model.evaluate_step_NN(support, query)

                gold_entitys = query['gold_entitys']

                score = res['score']
                span = res['span']
                pred = res['pred']

                pred_entitys_origin = self.get_entity_tuples_from_span(score, pred, span, self.predict_threshold)

                correct_cnt, pred_cnt, label_cnt, false_cnt, fp_span_cnt, fp_type_cnt, final_correct_span_cnt, \
                    final_pred_span_cnt = metrics_by_entity_tuples(gold_entitys, pred_entitys_origin)  # (span_l, span_r, pred), (span_l, span_r, pred)
                label_cnt, correct_span_cnt, pred_span_cnt, long_cnt, short_cnt, wo_overlap_cnt,\
                    share_begin_cnt, share_end_cnt = metrics_by_span_detector(gold_entitys, span)

                precision, recall, f1 = get_p_r_f1(correct_cnt, pred_cnt, label_cnt)

                span_precision, span_recall, span_f1 = get_p_r_f1(correct_span_cnt, pred_span_cnt, label_cnt)

                iter_precision += precision
                iter_recall += recall
                iter_f1 += f1

                all_correct_cnt += correct_cnt
                all_pred_cnt += pred_cnt
                all_gold_cnt += label_cnt
                all_fp_span_cnt += fp_span_cnt
                all_fp_type_cnt += fp_type_cnt
                all_correct_span_cnt += correct_span_cnt
                all_pred_span_cnt += pred_span_cnt
                all_false_cnt += false_cnt
                all_p, all_r, all_f1 = get_p_r_f1(all_correct_cnt, all_pred_cnt, all_gold_cnt)
                cur_metric_string = " No.{} episode, Precision:{:.5f}, Recall:{:.5f}, F1:{:.5f}, Span_P:{:.5f}, " \
                                    "Span_R:{:.5f},  Span_F1:{:.5f} Loss:{:.5f} All_P:{:.5f} All_R:{:.5f}, All_F1:{:.5f}".format(iter_sample, precision,
                                    recall, f1, span_precision, span_recall, span_f1, episode_loss, all_p, all_r, all_f1)
                sys.stdout.write(cur_metric_string + '\r')
                sys.stdout.flush()
                iter_sample += 1
        
            # reload best trained model for next test episode
            state_dict = checkpoint['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)

        all_p, all_r, all_f1 = get_p_r_f1(all_correct_cnt, all_pred_cnt, all_gold_cnt)
        all_span_p, all_span_r, all_span_f1 = get_p_r_f1(all_correct_span_cnt, all_pred_span_cnt, all_gold_cnt)
        all_fp_span, all_fp_type = get_fp(all_false_cnt, all_fp_span_cnt, all_fp_type_cnt)
        res_string = '''{} || [TEST] || F1 [FewNERD]: ( P: {:.4f}; R: {:.4f}; F1: {:.4f} FalseTotal:{}, FalseRation:{:.4f}, FP_span: {:.4f} FP_type: {:.4f} P_span: {:.4f}; R_span: {:.4f}; F1_span: {:.4f})'''.format(
            now(), all_p, all_r, all_f1, all_false_cnt, all_false_cnt / (all_pred_cnt + 1e-6),  all_fp_span, all_fp_type, all_span_p, all_span_r, all_span_f1)
        sys.stdout.write(res_string + '\r')
        sys.stdout.flush()
        print("")

        if ckpt is None:
            return iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample
        else:
            return "Test f1: {}; train_step: {} || ".format(dev_f1, train_step) + res_string



