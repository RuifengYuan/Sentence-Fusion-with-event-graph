# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:12:36 2020

@author: User
"""
import sys

sys.path.insert(0, 'G:/project/fusion_cnn/model')

sys.path.insert(0,'/home/ruifeng/project/fusion_cnn/model')

sys.path.insert(0,'/home/mist/data/model')  
  
import BERT_Encoder

import Decoder

import trans_decoder

import torch.nn as nn

import copy

class fact_model(nn.Module):

    def __init__(self,config):

        super(fact_model,self).__init__()  
      
        encoder_d = BERT_Encoder.encoder_both_fact()        
        

        encoder_d = encoder_d.cuda()
        
        if config.fp16 == 1:
            encoder_d = encoder_d.half()

        self.encoder_d = encoder_d
       
    def set_eval(self):
        
        self.encoder_d.eval()
        
    def set_train(self):
        self.encoder_d.train()

    def set_cuda(self):
        self.encoder_d.cuda()



class sum_model_bert_nopretrain(nn.Module):

    def __init__(self,config):
        
        super(sum_model_bert_nopretrain,self).__init__()  
        
        encoder_d = BERT_Encoder.encoder_both_nopretrain()  
          
        decoder = Decoder.LSTM_Decoder_new_x_share_emb(config, encoder_d.bert.embeddings.word_embeddings)
        
        encoder_d = encoder_d.cuda()

        decoder = decoder.cuda()

        if config.fp16 == 1:
            
            encoder_d = encoder_d.half()

            decoder = decoder.half()


        self.encoder_d = encoder_d

        self.decoder = decoder
        
        
    def set_eval(self):
        
        self.encoder_d.eval()

        self.decoder.eval()
        
    def set_train(self):
        
        self.encoder_d.train()

        self.decoder.train()
        
    def set_cuda(self):
        self.encoder_d.cuda()

        self.decoder.cuda()


class sum_model_bert(nn.Module):

    def __init__(self,config):
        
        super(sum_model_bert,self).__init__()  
        
        encoder_d = BERT_Encoder.encoder_both()  
          
        decoder = Decoder.LSTM_Decoder_new_x_share_emb(config, encoder_d.bert.embeddings.word_embeddings)
             
        encoder_d = encoder_d.cuda()

        decoder = decoder.cuda()

        if config.fp16 == 1:
            
            encoder_d = encoder_d.half()

            decoder = decoder.half()


        self.encoder_d = encoder_d

        self.decoder = decoder
        
        
    def set_eval(self):
        
        self.encoder_d.eval()

        self.decoder.eval()
        
    def set_train(self):
        
        self.encoder_d.train()

        self.decoder.train()
        
    def set_cuda(self):
        self.encoder_d.cuda()

        self.decoder.cuda()

class sum_model_bert_only_context(nn.Module):

    def __init__(self,config):
        
        super(sum_model_bert_only_context,self).__init__()  
        
        encoder_d = BERT_Encoder.encoder_both()  
          
        decoder = Decoder.LSTM_Decoder_new_x_share_emb_only_context(config, encoder_d.bert.embeddings.word_embeddings)
             
        encoder_d = encoder_d.cuda()

        decoder = decoder.cuda()

        if config.fp16 == 1:
            
            encoder_d = encoder_d.half()

            decoder = decoder.half()


        self.encoder_d = encoder_d

        self.decoder = decoder
        
        
    def set_eval(self):
        
        self.encoder_d.eval()

        self.decoder.eval()
        
    def set_train(self):
        
        self.encoder_d.train()

        self.decoder.train()
        
    def set_cuda(self):
        self.encoder_d.cuda()

        self.decoder.cuda()
        
class sum_model_bert_only_flow(nn.Module):

    def __init__(self,config):
        
        super(sum_model_bert_only_flow,self).__init__()  
        
        encoder_d = BERT_Encoder.encoder_both()  
          
        decoder = Decoder.LSTM_Decoder_new_x_share_emb_only_flow(config, encoder_d.bert.embeddings.word_embeddings)
             
        encoder_d = encoder_d.cuda()

        decoder = decoder.cuda()

        if config.fp16 == 1:
            
            encoder_d = encoder_d.half()

            decoder = decoder.half()


        self.encoder_d = encoder_d

        self.decoder = decoder
        
        
    def set_eval(self):
        
        self.encoder_d.eval()

        self.decoder.eval()
        
    def set_train(self):
        
        self.encoder_d.train()

        self.decoder.train()
        
    def set_cuda(self):
        self.encoder_d.cuda()

        self.decoder.cuda()


class sum_model_bert_GNN(nn.Module):

    def __init__(self,config):
        
        super(sum_model_bert_GNN,self).__init__()  
        
        encoder_d = BERT_Encoder.encoder_both()  
          
        decoder = Decoder.LSTM_Decoder_new_x_share_emb(config, encoder_d.bert.embeddings.word_embeddings)
        
        encoder_g = BERT_Encoder.encoder_gnn_ext()
             
        encoder_d = encoder_d.cuda()
        
        encoder_g = encoder_g.cuda()

        decoder = decoder.cuda()

        if config.fp16 == 1:
            
            encoder_d = encoder_d.half()

            decoder = decoder.half()


        self.encoder_d = encoder_d

        self.decoder = decoder
        
        self.encoder_g = encoder_g
        
        
    def set_eval(self):
        
        self.encoder_d.eval()

        self.decoder.eval()
        
        self.encoder_g.eval()
        
    def set_train(self):
        
        self.encoder_d.train()

        self.decoder.train()
        
        self.encoder_g.train()
        
    def set_cuda(self):
        self.encoder_d.cuda()

        self.decoder.cuda()
        
        self.encoder_g.cuda()