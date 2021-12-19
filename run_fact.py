
import sys

import os

import time

import argparse

from threading import Thread

import glob

import random

import pickle

import queue

import numpy as np

import data_fact_trans 

import torch

import torch.nn as nn

from model.Model import fact_model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adam

from transformers import BertTokenizer

from data_loader_fact import data_loader_train

from pytorch_pretrained_bert.optimization import BertAdam


class Batch(object):

  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list):
      
    self.abstract=[]
    self.article_res=[]
    self.article_no_res=[]
    self.graph=[]
    self.graph_meg=[]

    self.init_batch(example_list) # initialize the input to the encoder

        
  def init_batch(self, example_list):

    for ex in example_list:
        self.abstract.append(ex.abstract)
        self.article_res.append(ex.article_resolution)
        self.article_no_res.append(ex.article_no_resolution)
        self.graph.append(ex.graph_final)
        self.graph_meg.append(ex.graph_final_merged)
        

class Train(object):

    def __init__(self, config):
        
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.config = config  
        self.log = open('log.txt','w')
        self.trans=data_fact_trans.sample()
        self.batch_queue = queue.Queue(5)        

        self._batch_q_threads = []
        self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
        self._batch_q_threads[-1].daemon = True
        self._batch_q_threads[-1].start()
        
        

            
        
        if config.graph_only == 0 and config.doc_only == 0:
            if config.mid_start == 0:
                self.model = fact_model(config)
                self.model.set_cuda()
            else:
                x=torch.load('save_model/'+config.mid_model,map_location='cpu')
                self.model=x['model']
                self.model.set_cuda()
         
            no_change = ['bert.embeddings.word_embeddings.weight', 'embedding.weight']

            param_optimizer1 = list(self.model.encoder_d.named_parameters())
            
            
            optimizer_grouped_parameters=[
                    {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_change)], 'lr': config.lr_e},
                    {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_change)],'lr':config.lr_e}]
        
        self.optimizer = BertAdam(optimizer_grouped_parameters,lr=config.lr_e, e=1e-9)

        if config.fp16 == 1:
            from apex import amp
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
      
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
        
        model_save_path = 'save_model/'+str(self.iter)+'_iter_of_'+str(self.ecop) +'_ecop__rouge_'+str(rouge1)+'_'+str(rouge2)+'__loss_'+str(running_avg_loss)
        torch.save(state, model_save_path)


    def fill_batch_queue(self):

        self.iter=0
        
        self.ecop=0
        
        for i in range(50):
            
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
                
                for mini in range(int(16/self.config.mini_batch)):
                    try:    
                        article=one_batch['article'][start:start+self.config.mini_batch]
                        abstract=one_batch['abstract'][start:start+self.config.mini_batch]
                        graph=one_batch['graph'][start:start+self.config.mini_batch]  
                        
                        src,pos,seg,mask_d,pad_mask,labels,clss,methods= \
                        data_loader_train(article,abstract,self.tokenizer,self.config,self.trans)                  

                        data=[src,pos,seg,mask_d,pad_mask,labels,clss,methods]
                        self.batch_queue.put(data)                         
                    except Exception as e:
                        print('preprocess fail')
                        print(e)
                        pass
                    
                    start=start+self.config.mini_batch  
                        
                self.iter=self.iter+1
                
            self.ecop=self.ecop+1

                    
    def get_one_batch(self):
        
        data = self.batch_queue.get()     

        return data               


    def train_one_batch(self):
        
        src,pos,seg,mask_d,pad_mask,labels,clss,methods = self.get_one_batch()


        encoder_outputs_d,predicts = self.model.encoder_d(src,pos,seg,mask_d,clss)    

        
         
        
        ext_lossFunc = nn.BCEWithLogitsLoss()
        ext_lossFunc = ext_lossFunc.cuda()
        oracle_label=labels.type_as(predicts)
        ext_loss = torch.mean(ext_lossFunc(predicts, oracle_label))

        floss=ext_loss
        
        
        if torch.isnan(ext_loss) == True:
            return 0,0,0
           
        


        floss.backward()
                
        return floss.item(),floss.item(),1


    def trainIters(self, n_ecop):
        loss_list=[]
        e_loss_list=[]
        count=0
        
        self.model.set_train() 
        
        for i in range(n_ecop*1200):
            time_start=time.time()
            try:
                success=0
                for j in range(int(16/self.config.mini_batch)):
                                                  
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
    
                        
                if success == int(16/self.config.mini_batch):     
           
                    if self.config.graph_only == 0 and self.config.doc_only == 0:
                        clip_grad_norm_(self.model.encoder_d.parameters(), self.config.max_grad_norm)               
                    
    
                    self.optimizer.step()                         
                    self.optimizer.zero_grad()
    
    
                else:
                    print('jump one batch') 
                    
            except Exception as e:
                print('one batch fail')                  
                print('Reason for batch fail:', e)
                savedStdout = sys.stdout  #保存标准输出流
                with open('out.txt', 'a') as file:
                    sys.stdout = file  #标准输出重定向至文件
                    print('one batch fail')                  
                    print('Reason for batch fail:', e)
                sys.stdout = savedStdout  #恢复标准输出流                  
                
            time_end=time.time()
            
            if count % self.config.checkfreq == 0:
                len_loss=len(loss_list)
                if len_loss <=500*int(16/self.config.mini_batch):
                    recent_loss=loss_list[0:]
                    e_recent_loss=e_loss_list[0:]
                else:
                    recent_loss=loss_list[len_loss-500*int(16/self.config.mini_batch):]
                    e_recent_loss=e_loss_list[len_loss-500*int(16/self.config.mini_batch):]
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
                if len_loss <=500*int(16/self.config.mini_batch):
                    recent_loss=loss_list[0:]
                else:
                    recent_loss=loss_list[len_loss-500*int(16/self.config.mini_batch):]
                avg_loss=sum(recent_loss)/len(recent_loss)
                
                print('start val')
                acc=self.do_val(50)
                
                self.save_model(avg_loss,loss_list,acc,acc) 
                
                self.model.set_train()
                
            count=count+1
                    
    
    def do_val(self, val_num):

        self.model.set_eval()
        
        val_filelist = glob.glob(self.config.val_path)
      
        a=[]

                
        for batch_path in val_filelist[:val_num]:        
            
            f=open(batch_path,'rb')
            one_batch= pickle.load(f)
            start=0
            divide=1
            
            for mini in range(int(16/divide)):
                
#                try:
                    article=one_batch['article'][start:start+divide]
                    abstract=one_batch['abstract'][start:start+divide]
                    graph=one_batch['graph'][start:start+divide]    
    
                    start=start+divide
                    
                    acc = self.val_one_batch(article,abstract,graph)  
                    a.append(acc)
#                except:
#                    print('one sample batch fail')   
        if len(a) != 0:
            print(np.mean(a))
            return np.mean(a)
        else:
            return 0           
          
    def val_one_batch(self, article,abstract,graph):
        
        src,pos,seg,mask_d,pad_mask,labels,clss,methods= \
        data_loader_train(article,abstract, self.tokenizer, self.config, self.trans)


        encoder_outputs_d,predicts = self.model.encoder_d(src,pos,seg,mask_d,clss)    

        
        count=0

        c0=[]
        c1=[]
        c2=[]
        c3=[]
        c4=[]
        c5=[]
        
        for i in range(len(labels)):
            if methods[i] == 0:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c0.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c0.append(1)
                else:
                    c0.append(0)
            if methods[i] == 1:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c1.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c1.append(1)
                else:
                    c1.append(0)
            if methods[i] == 2:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c2.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c2.append(1)
                else:
                    c2.append(0)
            if methods[i] == 3:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c3.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c3.append(1)
                else:
                    c3.append(0)
            if methods[i] == 4:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c4.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c4.append(1)
                else:
                    c4.append(0)
            if methods[i] == 5:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c5.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c5.append(1)
                else:
                    c5.append(0)                    
                    
            if int(labels[i])==1 and predicts.item()>0.5:
                count=count+1
            if int(labels[i])==0 and predicts.item()<0.5:
                count=count+1
                
        return count/len(labels)        



class Test(object):
    
    def __init__(self, config):
        
        x=torch.load('save_model/'+config.test_model,map_location='cpu')
        
        self.config=config
        
        self.model=x['model']
        
        self.model.set_cuda()
        self.model.set_eval()
        
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.trans=data_fact_trans.sample()

    
    def test(self,test_num):
        self.model.set_eval()
        
        test_filelist = glob.glob(self.config.test_path)

        a=[]
        a0=[]
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        for batch_path in test_filelist[:test_num]:        
            f=open(batch_path,'rb')
            one_batch= pickle.load(f)
            start=0
            divide=1
            print('start one batch testing')
            
            for mini in range(int(16/divide)):
                
                try:
                    
                    article=one_batch['article'][start:start+divide]
                    abstract=one_batch['abstract'][start:start+divide]
                    graph=one_batch['graph'][start:start+divide]    
                    #print(article)
                    start=start+divide
                    
                    acc,c0,c1,c2,c3,c4,c5 = self.val_one_batch(article,abstract,graph)        
                    a.append(acc)
                    a0=a0+c0
                    a1=a1+c1
                    a2=a2+c2
                    a3=a3+c3
                    a4=a4+c4
                    a5=a5+c5
                except Exception as e:
                    print('one test batch fail')                  
                    print('Reason for batch fail:', e)

                    
        print(np.mean(a))
        print(sum(a0)/len(a0),len(a0))
        print(sum(a1)/len(a1),len(a1))
        print(sum(a2)/len(a2),len(a2))
        print(sum(a3)/len(a3),len(a3))
        print(sum(a4)/len(a4),len(a4))
        print(sum(a5)/len(a5),len(a5))
                          
        #rouges = test_rouge('result/rouge', self.can_path, self.gold_path)
        #print(rouge_results_to_str(rouges))


    def val_one_batch(self, article,abstract,graph):
        
        src,pos,seg,mask_d,pad_mask,labels,clss,methods= \
        data_loader_train(article,abstract, self.tokenizer, self.config, self.trans)


        encoder_outputs_d,predicts = self.model.encoder_d(src,pos,seg,mask_d,clss)    

        
        count=0

        c0=[]
        c1=[]
        c2=[]
        c3=[]
        c4=[]
        c5=[]
        
        for i in range(len(labels)):
            if methods[i] == 0:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c0.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c0.append(1)
                else:
                    c0.append(0)
            if methods[i] == 1:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c1.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c1.append(1)
                else:
                    c1.append(0)
            if methods[i] == 2:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c2.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c2.append(1)
                else:
                    c2.append(0)
            if methods[i] == 3:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c3.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c3.append(1)
                else:
                    c3.append(0)
            if methods[i] == 4:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c4.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c4.append(1)
                else:
                    c4.append(0)
            if methods[i] == 5:
                if int(labels[i])==1 and predicts.item()>0.5:
                    c5.append(1)
                elif int(labels[i])==0 and predicts.item()<0.5:
                    c5.append(1)
                else:
                    c5.append(0)                    
                    
            if int(labels[i])==1 and predicts.item()>0.5:
                count=count+1
            if int(labels[i])==0 and predicts.item()<0.5:
                count=count+1
                
        return count/len(labels),c0,c1,c2,c3,c4,c5  


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

    parser.add_argument('--lr_e', type=float, default=2e-06)     
    
    parser.add_argument('--lr_d', type=float, default=0.0001)     

    parser.add_argument('--loss_scale', type=float, default=0)  
    
    parser.add_argument('--eps', type=float, default=1e-10)
    
    parser.add_argument('--max_grad_norm', type=float, default=1)
        
    parser.add_argument('--mini_batch', type=int, default=8)  
    
    parser.add_argument('--use_hyp', type=int, default=1)
    
    # Init Setting
    
    parser.add_argument('--rand_unif_init_mag', type=float, default=0.02)
        
    parser.add_argument('--trunc_norm_init_std', type=float, default=1e-4)
    
    parser.add_argument('--adagrad_init_acc', type=float, default=0.1)
    
    # Data Setting

    parser.add_argument('--train_path', type=str, default='data_fact/multi/train/*')

    parser.add_argument('--val_path', type=str, default='data_fact/multi/val/*')

    parser.add_argument('--test_path', type=str, default='data_fact/multi/test/*')

    # Testing setting
    parser.add_argument('--beam_size', type=int, default=2)
    
    parser.add_argument('--max_dec_steps_val', type=int, default=60)
    
    parser.add_argument('--min_dec_steps', type=int, default=10)
    
    parser.add_argument('--test_model', type=str, default='')    
    # Checkpoint Setting

    parser.add_argument('--savefreq', type=int, default=1200)

    parser.add_argument('--checkfreq', type=int, default=1)
    
    # Model Setting

    parser.add_argument('--graph_only', type=int, default=0)     

    parser.add_argument('--doc_only', type=int, default=0)   

    parser.add_argument('--fp16', type=int, default=0)  
    
    parser.add_argument('--ext_node', type=int, default=1) 
    
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
        x.test(155)

main()
