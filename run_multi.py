# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:44:10 2020

@author: User
"""
import sys

sys.path.insert(0, 'G:/project/fusion_cnn/model')

sys.path.insert(0,'/home/ruifeng/project/fusion_cnn/model')

sys.path.insert(0,'/home/mist/data/')    

import os

import time

import argparse

import glob

import random

import pickle

import numpy as np

from rouge import Rouge

import torch

import torch.nn as nn

from model.Model import fact_model,sum_model_bert,sum_model_bert_only_context,sum_model_bert_only_flow

from torch.nn.utils import clip_grad_norm_

from torch.optim import Adam

from transformers import BertTokenizer

from data_loader import data_loader_train_split_new,data_loader_train_bert,fact_transform

from pytorch_pretrained_bert.optimization import BertAdam

from torch.optim.lr_scheduler import _LRScheduler




class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]



class Beam_fact(object):

  def __init__(self, tokens, log_probs, state, context, coverage, graph = None, coverage2 = None, flow = None, source = None, model=None, add_score=0):

    self.tokens = tokens

    self.log_probs = log_probs

    self.state = state

    self.context = context
    
    self.graph = graph

    self.coverage = coverage
    
    self.coverage2 = coverage2
    
    self.flow = flow

    self.source=source
    
    self.model=model
    
    self.add_score=add_score
    
    #print(self.add_score)
  def extend(self, token, log_prob, state, context, coverage, graph = None, coverage2 = None, flow = None, step=None):
      
     
     tar= self.tokens + [token]

     src,pos,seg,mask_d,pad_mask,clss=fact_transform(self.source,torch.tensor(tar))
     with torch.no_grad():      
         encoder_outputs_d,predicts = self.model.encoder_d(src,pos,seg,mask_d,clss) 

        
     ''' 
     fact_score=predicts[0]
     if step != 0 and token !=100:
         new_log_prob=log_prob+torch.clamp(torch.log(fact_score),-1,0)
     else:
         new_log_prob=log_prob
     '''
     
     new_log_prob=log_prob
     fact_score=predicts[0]     
     if step != 0 and token !=100:
         #self.add_score=torch.clamp(torch.log(fact_score),-0.5,0)
         self.add_score=torch.log(fact_score)*0.02
     else:
         self.add_score=0
     
     new_log_prob=log_prob+self.add_score  
     
     return Beam_fact(tokens = self.tokens + [token],

                      log_probs = self.log_probs + [new_log_prob],

                      state = state,

                      context = context,

                      coverage = coverage,
                      
                      graph = graph,
                      
                      coverage2 = coverage2,
                      
                      flow = flow,
                      
                      source=self.source,
                      
                      model=self.model,
                      
                      add_score=self.add_score)

  @property

  def latest_token(self):

    return self.tokens[-1]



  @property

  def avg_log_prob(self):

    #return sum(self.log_probs) / len(self.tokens) + self.add_score
    #print(sum(self.log_probs)/len(self.tokens), self.add_score/len(self.tokens))
    return sum(self.log_probs) / len(self.tokens)

class Beam(object):
    
  def __init__(self, tokens, log_probs, state, context, coverage, graph = None, coverage2 = None, flow = None):

    self.tokens = tokens

    self.log_probs = log_probs

    self.state = state

    self.context = context
    
    self.graph = graph

    self.coverage = coverage
    
    self.coverage2 = coverage2
    
    self.flow = flow



  def extend(self, token, log_prob, state, context, coverage, graph = None, coverage2 = None, flow = None,step=None):

    return Beam(tokens = self.tokens + [token],

                      log_probs = self.log_probs + [log_prob],

                      state = state,

                      context = context,

                      coverage = coverage,
                      
                      graph = graph,
                      
                      coverage2 = coverage2,
                      
                      flow = flow)

  @property

  def latest_token(self):

    return self.tokens[-1]



  @property

  def avg_log_prob(self):

    return sum(self.log_probs) / len(self.tokens)




class Train(object):

    def __init__(self, config):
        
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.config = config  
        self.log = open('log.txt','w')
        '''
        self.batch_queue = queue.Queue(5)        

        self._batch_q_threads = []
        self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
        self._batch_q_threads[-1].daemon = True
        self._batch_q_threads[-1].start()
        '''

        if config.mid_start == 0:
            self.model = sum_model_bert(config)
            self.model.set_cuda()
        else:
            x=torch.load('save_model_multi/'+config.mid_model,map_location='cpu')
            self.model=x['model']
            self.model.set_cuda()


        no_decay = ['bias', 'LayerNorm.weight']
        no_embeding = ['bert.embeddings.word_embeddings.weight', 'embedding.weight']

        param_optimizer1 = list(self.model.encoder_d.named_parameters())   
        param_optimizer2 = [(n,p) for n, p in list(self.model.decoder.named_parameters()) if not any(nd in n for nd in no_embeding)] 
        #param_optimizer1 = list(self.model.encoder_d.named_parameters()) 
        #param_optimizer2 = list()-list(self.model.decoder.named_parameters())

        optimizer_grouped_parameters=[
            {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,'lr':config.lr_e},
            {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':config.lr_e},
            {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001,'lr':config.lr_d},
            {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':config.lr_d}]
        
        self.optimizer = BertAdam(optimizer_grouped_parameters,lr=config.lr_e, e=1e-9)        
#        self.scheduler = NoamLR(self.optimizer, warmup_steps=6000*2)        
        self.next_data=self.fill_batch_queue()

        for i in self.optimizer.param_groups:
            print(i['lr'])

        #while(True):
            #pass
        
    def save_model(self, running_avg_loss,loss_list,rouge1,rouge2):

        state = {
            'iter': self.iter,
            'ecop': self.ecop,
            'model':self.model,
            'current_loss': running_avg_loss,
            'loss_list': loss_list,
            'rouge1':rouge1,
            'rouge2':rouge2            
        }
        
        model_save_path = 'save_model_multi_flow/'+str(self.iter)+'_iter_of_'+str(self.ecop) +'_ecop__rouge_'+str(rouge1)+'_'+str(rouge2)+'__loss_'+str(running_avg_loss)
        torch.save(state, model_save_path)


    def fill_batch_queue(self):

        self.iter=0
        
        self.ecop=0
        
        for i in range(100):
            
            train_filelist = glob.glob(self.config.train_path)
            
            random.shuffle(train_filelist)
            
            for batch_path in train_filelist:

                try:
                    f=open(batch_path,'rb')
                    one_batch= pickle.load(f)
                except:
                    print('pickle error')
                    continue
                
                start=0
                
                for mini in range(int(32/self.config.mini_batch)):
                    try:    
                        article=one_batch['article'][start:start+self.config.mini_batch]
                        abstract=one_batch['abstract'][start:start+self.config.mini_batch]
                        graph=one_batch['graph'][start:start+self.config.mini_batch]  
                        
                        src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d= \
                        data_loader_train_bert(article,abstract,graph,self.tokenizer,self.config)                  

                        data=[src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d]
                        #self.batch_queue.put(data)  
                        yield data
                    except Exception as e:
                        print('preprocess fail')
                        print(e)
                        pass
                    
                    start=start+self.config.mini_batch  
                        
                self.iter=self.iter+1
                
            self.ecop=self.ecop+1

                    
    def get_one_batch(self):
        
        data = self.next_data.__next__()

        return data               


    def train_one_batch(self):       
        src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d = \
        self.get_one_batch()
        if self.config.graph_only == 0 and self.config.doc_only == 0:

            encoder_outputs_d,encoder_outputs_g_n,predicts = self.model.encoder_d(src, pos,seg, mask_final,clsn,node_mask,clst,pad_mask_d)    
            B, tk, dim = encoder_outputs_d.size()            
            B, nk, tk = token_to_node.size()
            
             
            
            ext_lossFunc = nn.BCEWithLogitsLoss(weight=node_mask.float())
            ext_lossFunc = ext_lossFunc.cuda()
            oracle_label=oracle_label.type_as(predicts)
            ext_loss = torch.mean(ext_lossFunc(predicts, oracle_label))


            s_t_1 = (torch.zeros(1,len(src_d),self.config.hidden_dim, dtype=torch.float).cuda(),torch.zeros(1,len(src_d),self.config.hidden_dim, dtype=torch.float).cuda())   
            
            coverage=torch.zeros(src_d.size(),dtype=torch.float)            
            coverage=coverage.cuda()            
            flow=torch.zeros((B, nk),dtype=torch.float) + 1/nk           
            flow=flow.cuda()
            
            c_t_d=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()    
            c_t_g=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()
            c_t_g2=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda() 
            
            step_losses = []
            step_pg=[]
            step_l1=[]
            step_l2=[]
            step_l3=[]
            cov=[]
           
            for di in range(min(max_dec_len, self.config.max_dec_steps)):
    
                y_t_1 = inp[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_d, c_t_g,c_t_g2, attn_dist_d, attn_dist_g_node, p_gen, next_coverage, next_flow = self.model.decoder(y_t_1, s_t_1,
                                                            encoder_outputs_d, encoder_outputs_g_n, pad_mask_d, node_mask,
                                                            c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, flow_graph, src_d, di)
                
                #main loss
                target = tar[:, di]   
                step_mask = tar_mask[:, di]   
                
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()    
                step_loss = -torch.log(gold_probs + self.config.eps)
                step_l1.append(step_loss * step_mask) 
                
                #coverage_loss
                step_coverage_loss = torch.sum(torch.min(attn_dist_d, coverage), 1)    
                step_loss = step_loss + self.config.cov_loss_wt * step_coverage_loss    
                coverage = next_coverage
    
                #flow_loss
                '''
                loss_fn = torch.nn.L1Loss(reduction='none')
                step_flow_loss = torch.sum(loss_fn(attn_dist_g_node, flow), 1)              
                step_loss = step_loss + self.config.flow_loss_wt * step_flow_loss   
                '''
                flow = next_flow       
                flow_dis=attn_dist_g_node[1]+self.config.eps
                pred_dis=attn_dist_g_node[0]+self.config.eps
                #kl_loss
                loss_f_bs_mean = nn.KLDivLoss(reduction='batchmean')
                
                loss_bs_mean = loss_f_bs_mean(flow_dis.log(), pred_dis)

                step_loss = step_loss + loss_bs_mean * 0
                
                step_loss = step_loss * step_mask    
                step_losses.append(step_loss)

                step_pg.append(p_gen * step_mask)                
                step_l2.append(step_coverage_loss * step_mask)                
                step_l3.append(0 * step_mask)   
                cov.append(torch.max(next_coverage, 1))
    
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)

            
        batch_avg_loss = sum_losses/dec_len
        loss = torch.mean(batch_avg_loss)
        if self.config.ext_node == 1:
            floss=loss+ext_loss
            #floss=loss     
        step_l2_sum=torch.sum(torch.stack(step_l2, 1), 1)
        savedStdout = sys.stdout  #保存标准输出流
        with open('out.txt', 'a') as file:
            sys.stdout = file  #标准输出重定向至文件
            print(dec_len)
            print(torch.sum(torch.stack(step_l1, 1), 1))
            print(torch.sum(torch.stack(step_l2, 1), 1))
            print(torch.sum(torch.stack(step_l3, 1), 1))
            print(torch.sum(torch.stack(step_pg, 1), 1))
            print(loss)
            print(torch.max(step_l2_sum) - torch.min(step_l2_sum))          
            print('-------------')
        sys.stdout = savedStdout  #恢复标准输出流          
        
        
        #while(True):
            #pass
        
        
        for i in sum_losses:
            if torch.isnan(i) == True:
                return 0,0,0
        
        if torch.isnan(floss) == True:     
            return 0,0,0     
        
#        if float(torch.max(step_l2_sum) - torch.min(step_l2_sum)) > 10 * (self.config.max_dec_steps/100):
#            return 0,-1 
        

        if self.config.ext_node == 0:
            loss.backward()
        else:
            floss.backward()
                
        if self.config.ext_node == 0:
            return loss.item(),0,1
        else:             
            #print(loss.item(),ext_loss.item())
            return loss.item(),ext_loss.item(),1


    def trainIters(self, n_ecop):
        loss_list=[0]
        e_loss_list=[0]
        count=0
        
        self.model.set_train() 
        
        for i in range(n_ecop*630):
            time_start=time.time()
            try:
                success=0
                for j in range(int(32/self.config.mini_batch)):
                                                  
                    loss,eloss,tag = self.train_one_batch()
                    if tag == 1:
                        loss_list.append(loss)
                        e_loss_list.append(eloss)
                        success=success+1
                    if tag == 0:
                        print('one mini batch fail for NAN')                            
                        continue
                    if tag == -1:
                        print('one mini batch fail for COV')                            
                        continue
    
                        
                if success == int(32/self.config.mini_batch):     
                    
                    #for name, parms in self.model.encoder_g.named_parameters():	
                        #print('-->name:', name, '-->grad_requirs:', parms.requires_grad, 'rad_value:',parms.grad)
                    
                    
                    if self.config.graph_only == 0 and self.config.doc_only == 0:
                        clip_grad_norm_(self.model.encoder_d.parameters(), self.config.max_grad_norm)               
                        clip_grad_norm_(self.model.decoder.parameters(), self.config.max_grad_norm)
                    
    
                    self.optimizer.step()                         
                    self.optimizer.zero_grad()
#                    self.scheduler.step() 
    
                else:
                    print('jump one batch') 
                    
                    
            except Exception as e:
                print('one batch fail')                  
                print('Reason for batch fail:', e)
#                savedStdout = sys.stdout  #保存标准输出流
#                with open('out.txt', 'a') as file:
#                    sys.stdout = file  #标准输出重定向至文件
#                    print('one batch fail')                  
#                    print('Reason for batch fail:', e)
#                sys.stdout = savedStdout  #恢复标准输出流                  
                
            time_end=time.time()
            
            if count % self.config.checkfreq == 0:
                len_loss=len(loss_list)
                if len_loss <=500*int(32/self.config.mini_batch):
                    recent_loss=loss_list[0:]
                    e_recent_loss=e_loss_list[0:]
                else:
                    recent_loss=loss_list[len_loss-500*int(32/self.config.mini_batch):]
                    e_recent_loss=e_loss_list[len_loss-500*int(32/self.config.mini_batch):]
                avg_loss=sum(recent_loss)/len(recent_loss)
                e_avg_loss=sum(e_recent_loss)/len(e_recent_loss)
                savedStdout = sys.stdout  #保存标准输出流
                with open('out.txt', 'a') as file:
                    sys.stdout = file  #标准输出重定向至文件
                    print('-------------')
                    print(str(count)+' iter '+str(self.ecop) +' of ecop avg_loss:'+str(avg_loss)+' eloss:'+str(e_avg_loss)+' time:'+str(time_end-time_start))
                    print('-------------')
                sys.stdout = savedStdout  #恢复标准输出流 
                
                print(str(count)+' iter '+str(self.ecop) +' of ecop avg_loss:'+str(avg_loss)+' eloss:'+str(e_avg_loss)+' time:'+str(time_end-time_start))
                
            if count % self.config.savefreq == 0 and count > self.config.savefreq-100:     
                len_loss=len(loss_list)
                if len_loss <=500*int(32/self.config.mini_batch):
                    recent_loss=loss_list[0:]
                else:
                    recent_loss=loss_list[len_loss-500*int(32/self.config.mini_batch):]
                avg_loss=sum(recent_loss)/len(recent_loss)
                
                print('start val')
                rouge1,rouge2=self.do_val(50)
                
                self.save_model(avg_loss,loss_list,rouge1,rouge2) 
                
                self.model.set_train()
                
            count=count+1
                    
    
    def do_val(self, val_num):

        self.raw_rouge=Rouge()
        
        self.model.set_eval()
        
        val_filelist = glob.glob(self.config.val_path)
      
        r1=[]
        r2=[]
                
        for batch_path in val_filelist[:val_num]:        
            
            f=open(batch_path,'rb')
            one_batch= pickle.load(f)
            start=0
            divide=1

            for mini in range(int(32/divide)):
                
                try:
                    article=one_batch['article'][start:start+divide]
                    abstract=one_batch['abstract'][start:start+divide]
                    graph=one_batch['graph'][start:start+divide]    
    
                    start=start+divide
                    
                    best = self.beam_search(article,abstract,graph)
                    pred = self.tokenizer.decode(best.tokens)
                    pred = pred.replace(' [SEP] ', ' <q> ')
                    gold = ''

                    ii=self.tokenizer.encode(abstract[0],add_special_tokens=False)
                    ii=self.tokenizer.decode(ii)
                    gold=gold+ii+' [UNK] '
                        
                    pred=pred.encode('ascii', errors='replace')    
                    gold=gold.encode('ascii', errors='replace')   
                    pred=pred.decode('ascii')    
                    gold=gold.decode('ascii')  
                       
                    scores = self.raw_rouge.get_scores(pred, gold)

                    r1.append(scores[0]['rouge-1']['f'])
                    r2.append(scores[0]['rouge-2']['f'])    
                    
                except:
                    print('one sample batch fail')   
        if len(r1) != 0 and len(r2) != 0:
            print(np.mean(r1),np.mean(r2))
            return np.mean(r1),np.mean(r2)
        else:
            return 0,0              
          
            
    def beam_search(self,article,abstract,graph):
        
        src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d = \
        data_loader_train_bert(article,abstract,graph,self.tokenizer,self.config)  
        if self.config.beam_size == 4: 
            src_d=torch.cat((src_d,src_d,src_d,src_d),0)
            src=torch.cat((src,src,src,src),0)
            pos=torch.cat((pos,pos,pos,pos),0)
            seg=torch.cat((seg,seg,seg,seg),0)
            mask_final=torch.cat((mask_final,mask_final,mask_final,mask_final),0)
            clst=torch.cat((clst,clst,clst,clst),0)  
            clsn=torch.cat((clsn,clsn,clsn,clsn),0)             
            
            pad_mask_d=torch.cat((pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d),0)      

            token_to_node=torch.cat((token_to_node,token_to_node,token_to_node,token_to_node),0)   
            node_mask=torch.cat((node_mask,node_mask,node_mask,node_mask),0)   
            flow_graph=torch.cat((flow_graph,flow_graph,flow_graph,flow_graph),0) 

            node_to_token=torch.cat((node_to_token,node_to_token,node_to_token,node_to_token),0)        
            inp=torch.cat((inp,inp,inp,inp),0)
            tar=torch.cat((tar,tar,tar,tar),0)
            tar_mask=torch.cat((tar_mask,tar_mask,tar_mask,tar_mask),0)
            dec_len=torch.cat((dec_len,dec_len,dec_len,dec_len),0)
            adj_graph=torch.cat((adj_graph,adj_graph,adj_graph,adj_graph),0)

        if self.config.beam_size == 2:  
            src_d=torch.cat((src_d,src_d),0)
            src=torch.cat((src,src),0)
            pos=torch.cat((pos,pos),0)
            seg=torch.cat((seg,seg),0)
            mask_final=torch.cat((mask_final,mask_final),0)
            clst=torch.cat((clst,clst),0)  
            clsn=torch.cat((clsn,clsn),0)   
            
            pad_mask_d=torch.cat((pad_mask_d,pad_mask_d),0)      

            token_to_node=torch.cat((token_to_node,token_to_node),0)   
            node_mask=torch.cat((node_mask,node_mask),0)   
            flow_graph=torch.cat((flow_graph,flow_graph),0) 


            node_to_token=torch.cat((node_to_token,node_to_token),0)        
            inp=torch.cat((inp,inp),0)
            tar=torch.cat((tar,tar),0)
            tar_mask=torch.cat((tar_mask,tar_mask),0)
            dec_len=torch.cat((dec_len,dec_len),0)
            adj_graph=torch.cat((adj_graph,adj_graph),0)
  
        if self.config.graph_only == 0 and self.config.doc_only == 0:
           
            encoder_outputs_d,encoder_outputs_g_n,_ = self.model.encoder_d(src, pos,seg, mask_final,clsn,node_mask,clst,pad_mask_d)    

            B, tk, dim = encoder_outputs_d.size()            
            B, nk, tk = token_to_node.size()  

            s_t_1 = (torch.zeros(1,1,self.config.hidden_dim, dtype=torch.float).cuda(),torch.zeros(1,1,self.config.hidden_dim, dtype=torch.float).cuda())   
            
            coverage=torch.zeros(src_d.size(),dtype=torch.float)            
            coverage=coverage.cuda()            
            flow=torch.zeros((B, nk),dtype=torch.float)            
            flow=flow.cuda()
            
            c_t_d=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()    
            c_t_g=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()
            c_t_g2=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()
            
            dec_h, dec_c = s_t_1 # 1 x 2*hidden_size    
            dec_h = dec_h.squeeze()    
            dec_c = dec_c.squeeze()            
            
            beams = [Beam(tokens=[102],    
                          log_probs=[0.0],    
                          state=(dec_h, dec_c),    
                          context = c_t_d[0],    
                          coverage=coverage[0], 
                          graph = c_t_g[0],
                          coverage2=c_t_g2[0],
                          flow=flow[0])
    
                     for _ in range(self.config.beam_size)]
                        
            results = []
            steps = 0
            focus_node=[]
            while steps < self.config.max_dec_steps_val and len(results) < self.config.beam_size: 

                latest_tokens = [h.latest_token for h in beams]    
    
                y_t_1 = torch.LongTensor(latest_tokens)

                y_t_1 = y_t_1.cuda()
    
                all_state_h =[]
                all_state_c = []   
                all_context = []
                all_coverage = []
                all_graph=[]
                all_coverage2=[]
                all_flow = []
                for h in beams:    
                    state_h, state_c = h.state    
                    all_state_h.append(state_h)    
                    all_state_c.append(state_c)    
                    all_context.append(h.context)
                    all_coverage.append(h.coverage)
                    all_graph.append(h.graph)
                    all_coverage2.append(h.coverage2)
                    all_flow.append(h.flow)
                s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))    
                c_t_d = torch.stack(all_context, 0) 
                c_t_g = torch.stack(all_graph, 0)
                c_t_g2 = torch.stack(all_coverage2, 0)
                coverage = torch.stack(all_coverage, 0)              
                flow = torch.stack(all_flow, 0)    

                final_dist, s_t_1, c_t_d, c_t_g,c_t_g2, attn_dist_d, attn_dist_g_node, p_gen, next_coverage, next_flow = self.model.decoder(y_t_1, s_t_1,
                                                            encoder_outputs_d, encoder_outputs_g_n, pad_mask_d, node_mask,
                                                            c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, flow_graph, src_d, steps)

                
                log_probs = torch.log(final_dist)
                
                topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)

                dec_h, dec_c = s_t_1   
                dec_h = dec_h.squeeze()
                dec_c = dec_c.squeeze()

        
                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t_d[i]
                    coverage_i = next_coverage[i] 
                    graph_i = c_t_g[i]   
                    coverage2_i=c_t_g2[i] 
                    flow_i = next_flow[i]
    
                    for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                        new_beam = h.extend(token=topk_ids[i, j].item(),
                                       log_prob=topk_log_probs[i, j].item(),
                                       state=state_i,
                                       context=context_i,
                                       coverage=coverage_i,
                                       graph=graph_i,
                                       coverage2=coverage2_i,
                                       flow=flow_i,
                                       step=steps)
    
                        all_beams.append(new_beam)
    
    
    
                beams = []
                for h in self.sort_beams(all_beams):
                    if h.latest_token == 100:
                        if steps >= self.config.min_dec_steps:
                            results.append(h)
                    else:
                        beams.append(h)
                    if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                        break
                steps=steps+1                                                                                                         
            

            
            if len(results) == 0:    
                results = beams        
            beams_sorted = self.sort_beams(results)    
            return beams_sorted[0]        

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True) 



class Test(object):
    
    def __init__(self, config):
        
        x=torch.load('save_model_multi/'+config.test_model,map_location='cpu')
        
        self.config=config
        
        self.model=x['model']
        
        self.model.set_cuda()
        self.model.set_eval()
        if self.config.beam_with_fact==1:
            y=torch.load('fact_model/'+config.fact_model,map_location='cpu')
            self.fact_model=y['model']
            self.fact_model.set_cuda()
            self.fact_model.set_eval()
        
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.raw_rouge=Rouge()
        
        self.can_path = 'result/'+config.test_model+'_cand.txt'

        self.gold_path ='result/'+config.test_model+'_gold.txt'
        
        self.source_path ='result/'+config.test_model+'_source.txt'
    def beam_search(self,article,abstract,graph):
        
        src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d = \
        data_loader_train_bert(article,abstract,graph,self.tokenizer,self.config)  

        if self.config.beam_size == 5: 
            src_d=torch.cat((src_d,src_d,src_d,src_d,src_d),0)
            src=torch.cat((src,src,src,src,src),0)
            pos=torch.cat((pos,pos,pos,pos,pos),0)
            seg=torch.cat((seg,seg,seg,seg,seg),0)
            mask_final=torch.cat((mask_final,mask_final,mask_final,mask_final,mask_final),0)
            clst=torch.cat((clst,clst,clst,clst,clst),0)  
            clsn=torch.cat((clsn,clsn,clsn,clsn,clsn),0)             
            
            pad_mask_d=torch.cat((pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d),0)      

            token_to_node=torch.cat((token_to_node,token_to_node,token_to_node,token_to_node,token_to_node),0)   
            node_mask=torch.cat((node_mask,node_mask,node_mask,node_mask,node_mask),0)   
            flow_graph=torch.cat((flow_graph,flow_graph,flow_graph,flow_graph,flow_graph),0) 

            node_to_token=torch.cat((node_to_token,node_to_token,node_to_token,node_to_token,node_to_token),0)        
            inp=torch.cat((inp,inp,inp,inp,inp),0)
            tar=torch.cat((tar,tar,tar,tar,tar),0)
            tar_mask=torch.cat((tar_mask,tar_mask,tar_mask,tar_mask,tar_mask),0)
            dec_len=torch.cat((dec_len,dec_len,dec_len,dec_len,dec_len),0)
            adj_graph=torch.cat((adj_graph,adj_graph,adj_graph,adj_graph,adj_graph),0)
        
        if self.config.beam_size == 6: 
            src_d=torch.cat((src_d,src_d,src_d,src_d,src_d,src_d),0)
            src=torch.cat((src,src,src,src,src,src),0)
            pos=torch.cat((pos,pos,pos,pos,pos,pos),0)
            seg=torch.cat((seg,seg,seg,seg,seg,seg),0)
            mask_final=torch.cat((mask_final,mask_final,mask_final,mask_final,mask_final,mask_final),0)
            clst=torch.cat((clst,clst,clst,clst,clst,clst),0)  
            clsn=torch.cat((clsn,clsn,clsn,clsn,clsn,clsn),0)             
            
            pad_mask_d=torch.cat((pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d),0)      

            token_to_node=torch.cat((token_to_node,token_to_node,token_to_node,token_to_node,token_to_node,token_to_node),0)   
            node_mask=torch.cat((node_mask,node_mask,node_mask,node_mask,node_mask,node_mask),0)   
            flow_graph=torch.cat((flow_graph,flow_graph,flow_graph,flow_graph,flow_graph,flow_graph),0) 

            node_to_token=torch.cat((node_to_token,node_to_token,node_to_token,node_to_token,node_to_token,node_to_token),0)        
            inp=torch.cat((inp,inp,inp,inp,inp,inp),0)
            tar=torch.cat((tar,tar,tar,tar,tar,tar),0)
            tar_mask=torch.cat((tar_mask,tar_mask,tar_mask,tar_mask,tar_mask,tar_mask),0)
            dec_len=torch.cat((dec_len,dec_len,dec_len,dec_len,dec_len,dec_len),0)
            adj_graph=torch.cat((adj_graph,adj_graph,adj_graph,adj_graph,adj_graph,adj_graph),0)
            
            
        if self.config.beam_size == 4: 
            src_d=torch.cat((src_d,src_d,src_d,src_d),0)
            src=torch.cat((src,src,src,src),0)
            pos=torch.cat((pos,pos,pos,pos),0)
            seg=torch.cat((seg,seg,seg,seg),0)
            mask_final=torch.cat((mask_final,mask_final,mask_final,mask_final),0)
            clst=torch.cat((clst,clst,clst,clst),0)  
            clsn=torch.cat((clsn,clsn,clsn,clsn),0)             
            
            pad_mask_d=torch.cat((pad_mask_d,pad_mask_d,pad_mask_d,pad_mask_d),0)      

            token_to_node=torch.cat((token_to_node,token_to_node,token_to_node,token_to_node),0)   
            node_mask=torch.cat((node_mask,node_mask,node_mask,node_mask),0)   
            flow_graph=torch.cat((flow_graph,flow_graph,flow_graph,flow_graph),0) 

            node_to_token=torch.cat((node_to_token,node_to_token,node_to_token,node_to_token),0)        
            inp=torch.cat((inp,inp,inp,inp),0)
            tar=torch.cat((tar,tar,tar,tar),0)
            tar_mask=torch.cat((tar_mask,tar_mask,tar_mask,tar_mask),0)
            dec_len=torch.cat((dec_len,dec_len,dec_len,dec_len),0)
            adj_graph=torch.cat((adj_graph,adj_graph,adj_graph,adj_graph),0)

        if self.config.beam_size == 2:  
            src_d=torch.cat((src_d,src_d),0)
            src=torch.cat((src,src),0)
            pos=torch.cat((pos,pos),0)
            seg=torch.cat((seg,seg),0)
            mask_final=torch.cat((mask_final,mask_final),0)
            clst=torch.cat((clst,clst),0)  
            clsn=torch.cat((clsn,clsn),0)   
            
            pad_mask_d=torch.cat((pad_mask_d,pad_mask_d),0)      

            token_to_node=torch.cat((token_to_node,token_to_node),0)   
            node_mask=torch.cat((node_mask,node_mask),0)   
            flow_graph=torch.cat((flow_graph,flow_graph),0) 


            node_to_token=torch.cat((node_to_token,node_to_token),0)        
            inp=torch.cat((inp,inp),0)
            tar=torch.cat((tar,tar),0)
            tar_mask=torch.cat((tar_mask,tar_mask),0)
            dec_len=torch.cat((dec_len,dec_len),0)
            adj_graph=torch.cat((adj_graph,adj_graph),0)
  
        if self.config.graph_only == 0 and self.config.doc_only == 0:
           
            encoder_outputs_d,encoder_outputs_g_n,_ = self.model.encoder_d(src, pos,seg, mask_final,clsn,node_mask,clst,pad_mask_d)    

            B, tk, dim = encoder_outputs_d.size()            
            B, nk, tk = token_to_node.size() 
            
            
            s_t_1 = (torch.zeros(1,1,self.config.hidden_dim, dtype=torch.float).cuda(),torch.zeros(1,1,self.config.hidden_dim, dtype=torch.float).cuda())   
            
            coverage=torch.zeros(src_d.size(),dtype=torch.float)            
            coverage=coverage.cuda()            
            flow=torch.zeros((B, nk),dtype=torch.float)            
            flow=flow.cuda()
            
            c_t_d=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()    
            c_t_g=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()
            c_t_g2=torch.zeros(len(src_d),self.config.hidden_dim, dtype=torch.float).cuda()
            
            dec_h, dec_c = s_t_1 # 1 x 2*hidden_size    
            dec_h = dec_h.squeeze()    
            dec_c = dec_c.squeeze()            
            if self.config.beam_with_fact==1:
                beams = [Beam_fact(tokens=[102],    
                              log_probs=[0.0],    
                              state=(dec_h, dec_c),    
                              context = c_t_d[0],    
                              coverage=coverage[0], 
                              graph = c_t_g[0],
                              coverage2=c_t_g2[0],
                              flow=flow[0],
                              source=src_d[0],
                              model=self.fact_model)
        
                         for _ in range(self.config.beam_size)]
            else:
                beams = [Beam(tokens=[102],    
                              log_probs=[0.0],    
                              state=(dec_h, dec_c),    
                              context = c_t_d[0],    
                              coverage=coverage[0], 
                              graph = c_t_g[0],
                              coverage2=c_t_g2[0],
                              flow=flow[0])
        
                         for _ in range(self.config.beam_size)]     
                   
            results = []
            steps = 0
            focus_node=[]
            
            while steps < self.config.max_dec_steps_val and len(results) < self.config.beam_size: 

                latest_tokens = [h.latest_token for h in beams]    
    
                y_t_1 = torch.LongTensor(latest_tokens)

                y_t_1 = y_t_1.cuda()
    
                all_state_h =[]
                all_state_c = []   
                all_context = []
                all_coverage = []
                all_graph=[]
                all_coverage2=[]
                all_flow = []
                for h in beams:    
                    state_h, state_c = h.state    
                    all_state_h.append(state_h)    
                    all_state_c.append(state_c)    
                    all_context.append(h.context)
                    all_coverage.append(h.coverage)
                    all_graph.append(h.graph)
                    all_coverage2.append(h.coverage2)
                    all_flow.append(h.flow)
                s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))    
                c_t_d = torch.stack(all_context, 0) 
                c_t_g = torch.stack(all_graph, 0)
                c_t_g2 = torch.stack(all_coverage2, 0)
                coverage = torch.stack(all_coverage, 0)              
                flow = torch.stack(all_flow, 0)    

                final_dist, s_t_1, c_t_d, c_t_g,c_t_g2, attn_dist_d, attn_dist_g_node, p_gen, next_coverage, next_flow = self.model.decoder(y_t_1, s_t_1,
                                                            encoder_outputs_d, encoder_outputs_g_n, pad_mask_d, node_mask,
                                                            c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, flow_graph, src_d, steps)

                #print(torch.topk(attn_dist_g_node[0], 5))
                #print(attn_dist_g_node[0])
                focus_node=focus_node+[int(kk) for kk in list(torch.topk(attn_dist_g_node[0], 1).indices)]
                '''
                xx = [int(kk) for kk in list(torch.topk(attn_dist_g_node[0], 30).indices)]
                tag=0
                for one in xx:
                    if one not in focus_node:
                        focus_node.append(one)
                        tag=1
                        break
                    
                if tag == 0:
                    focus_node.append(-1)                    
                '''                
                log_probs = torch.log(final_dist)
                
                topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)
                
                #print(self.tokenizer.decode(topk_ids[0]))
                #print('--------------------------------')
                
                dec_h, dec_c = s_t_1   
                dec_h = dec_h.squeeze()
                dec_c = dec_c.squeeze()

        
                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t_d[i]
                    coverage_i = next_coverage[i] 
                    graph_i = c_t_g[i]   
                    coverage2_i=c_t_g2[i] 
                    flow_i = next_flow[i]
  
                    for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:

                        new_beam = h.extend(token=topk_ids[i, j].item(),
                                       log_prob=topk_log_probs[i, j].item(),
                                       state=state_i,
                                       context=context_i,
                                       coverage=coverage_i,
                                       graph=graph_i,
                                       coverage2=coverage2_i,
                                       flow=flow_i,
                                       step=steps)
    
                        all_beams.append(new_beam)
    
    
                beams = []
                for h in self.sort_beams(all_beams):
                    if h.latest_token == 100:
                        if steps >= self.config.min_dec_steps:
                            results.append(h)
                    else:
                        beams.append(h)
                    if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                        break
                steps=steps+1                                                                                                         

            #print(focus_node)
            #print(len(list(set(focus_node))))

            
            if len(results) == 0:    
                results = beams        
            beams_sorted = self.sort_beams(results)    
            return beams_sorted[0]

    
    
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)    
    
    def test(self,test_num):
        self.model.set_eval()
        
        test_filelist = glob.glob(self.config.test_path)
        pred_list=[]
        gold_list=[]
        source_list=[]
        with open(self.can_path, 'w', encoding='utf-8') as save_pred:
            with open(self.gold_path, 'w', encoding='utf-8') as save_gold:
                with open(self.source_path, 'w', encoding='utf-8') as save_source:
                
                    for batch_path in test_filelist[:test_num]:        
                        f=open(batch_path,'rb')
                        one_batch= pickle.load(f)
                        start=0
                        divide=1
                        print('start one batch testing')
                        for mini in range(int(32/divide)):
                            
                            try:

                                article=one_batch['article'][start:start+divide]
                                abstract=one_batch['abstract'][start:start+divide]
                                graph=one_batch['graph'][start:start+divide]    

                                #print(article)
                                start=start+divide
                                
                                best = self.beam_search(article,abstract,graph)
                                pred = self.tokenizer.decode(best.tokens)
                                pred = pred.replace('[SEP]', '')
                                pred = pred.replace('[UNK]', '')      
                                
                                pred = pred.replace('.', ' .')                                
                                pred = pred.replace(',', ' ,') 
                                pred = pred.replace("'s", " 's") 
                                pred = pred.replace("n't", " n't") 
                                
                                gold = ''
    
                                ii=self.tokenizer.encode(abstract[0],add_special_tokens=False)
                                ii=self.tokenizer.decode(ii)
                                gold=gold+ii
                                
                                gold = gold.replace('.', ' .')                                
                                gold = gold.replace(',', ' ,') 
                                gold = gold.replace("'s", " 's") 
                                gold = gold.replace("n't", " n't") 
                                    
                                pred_list.append(pred)
                                gold_list.append(gold)      

                             
                                art=''
                                for s in article[0]:
                                    art=art+s.replace('\n','')+' . '
                                    
                                source_list.append(art) 
                                
                            except Exception as e:
                                print('one test batch fail')                  
                                print('Reason for batch fail:', e)

                    
                    for sent in gold_list:
                        save_gold.write(sent.strip() + '\n')
                    for sent in pred_list:
                        save_pred.write(sent.strip() + '\n')
                    for sent in source_list:
                        save_source.write(sent.strip() + '\n')
                          
        #rouges = test_rouge('result/rouge', self.can_path, self.gold_path)
        #print(rouge_results_to_str(rouges))




def argLoader():

    parser = argparse.ArgumentParser()
    
    
    #device
    
    parser.add_argument('--device', type=int, default=0)    
    
    # Do What
    
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")

    parser.add_argument('--do_test', action='store_true', help="Whether to run test")

    #Preprocess Setting
    parser.add_argument('--max_len', type=int, default=150)

    parser.add_argument('--max_enc_graph_len', type=int, default=400)    
    
    parser.add_argument('--min_sent_len', type=int, default=5)

    parser.add_argument('--max_node_len', type=int, default=16)

    parser.add_argument('--max_dec_steps', type=int, default=100)
    
    parser.add_argument('--add_root', type=int, default=0)  
    
    parser.add_argument('--one_direction_reduction', type=float, default=0)    
    
    #Model Setting
    parser.add_argument('--pos_dim', type=int, default=150)

    parser.add_argument('--hidden_dim', type=int, default=768)

    parser.add_argument('--emb_dim', type=int, default=768)
    
    parser.add_argument('--vocab_size', type=int, default=30522)    
    
    parser.add_argument('--cov_loss_wt', type=float, default=1)      

    parser.add_argument('--flow_loss_wt', type=float, default=0)     

    parser.add_argument('--lr_e', type=float, default=1e-05)     
    
    parser.add_argument('--lr_d', type=float, default=0.0005)     

    parser.add_argument('--loss_scale', type=float, default=0)  
    
    parser.add_argument('--eps', type=float, default=1e-10)
    
    parser.add_argument('--max_grad_norm', type=float, default=1)
        
    parser.add_argument('--mini_batch', type=int, default=8)  
    
    parser.add_argument('--true_batch', type=int, default=16) 
    
    parser.add_argument('--use_hyp', type=int, default=1)
    
    # Init Setting
    
    parser.add_argument('--rand_unif_init_mag', type=float, default=0.02)
        
    parser.add_argument('--trunc_norm_init_std', type=float, default=1e-4)
    
    parser.add_argument('--adagrad_init_acc', type=float, default=0.1)
    
    # Data Setting

    parser.add_argument('--train_path', type=str, default='data_file/multi/train/*')

    parser.add_argument('--val_path', type=str, default='data_file/multi/val/*')

    parser.add_argument('--test_path', type=str, default='data_file/multi/test/*')

    # Testing setting
    parser.add_argument('--beam_size', type=int, default=2)
    
    parser.add_argument('--max_dec_steps_val', type=int, default=60)
    
    parser.add_argument('--min_dec_steps', type=int, default=10)
    
    parser.add_argument('--test_model', type=str, default='')    
    
    parser.add_argument('--fact_model', type=str, default='')    
    # Checkpoint Setting

    parser.add_argument('--savefreq', type=int, default=600)

    parser.add_argument('--checkfreq', type=int, default=1)
    
    # Model Setting

    parser.add_argument('--graph_only', type=int, default=0)     

    parser.add_argument('--doc_only', type=int, default=0)   

    parser.add_argument('--fp16', type=int, default=0)  
    
    parser.add_argument('--ext_node', type=int, default=1) 
    
    parser.add_argument('--beam_with_fact', type=int, default=1) 
    
    #Mid start
    
    parser.add_argument('--mid_start', type=int, default=0) 

    parser.add_argument('--mid_model', type=str, default='')            
    
    args = parser.parse_args()
    
    return args





def main():

    args = argLoader()

    torch.cuda.set_device(args.device)

    print('CUDA', torch.cuda.current_device())
    

    if args.do_train:
        
        x=Train(args)
    
        x.trainIters(20)
        
    if args.do_test:
        
        x = Test(args)
        x.test(300)

main()
