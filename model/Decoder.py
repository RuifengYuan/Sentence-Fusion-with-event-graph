# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:44:45 2020

@author: User
"""
import torch

import torch.nn as nn

import torch.nn.functional as F

from Attention import *


def init_lstm_wt(lstm,config):

    for names in lstm._all_weights:

        for name in names:

            if name.startswith('weight_'):

                wt = getattr(lstm, name)

                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

            elif name.startswith('bias_'):

                # set forget bias to 1

                bias = getattr(lstm, name)

                n = bias.size(0)

                start, end = n // 4, n // 2

                bias.data.fill_(0.)

                bias.data[start:end].fill_(1.)



def init_linear_wt(linear,config):

    linear.weight.data.normal_(std=config.trunc_norm_init_std)

    if linear.bias is not None:

        linear.bias.data.normal_(std=config.trunc_norm_init_std)



    
class LSTM_Decoder_new_x(nn.Module):

    def __init__(self, config):

        super(LSTM_Decoder_new_x, self).__init__()

        self.config=config

        self.attention_d = Doc_Attention(config)
        
        self.attention_f = Flow_Attention_only_node_improve_x(config)

        # decoder

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        #
        self.x_context = nn.Linear(config.hidden_dim * 4, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        #
        self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)

        self.d_gra_linear = nn.Linear(config.hidden_dim * 4, 1)
        
        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)




    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_outputs_node, enc_padding_mask, enc_padding_mask_node,

                c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, graph, voc_d, step):

        b, tk, hid=encoder_outputs.size()
        b, nk, hid=encoder_outputs_node.size()

        y_t_1_embd = self.embedding(y_t_1)

        x = self.x_context(torch.cat((c_t_d,c_t_g, c_t_g2, y_t_1_embd), 1))
        
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t

        s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),

                             c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t_d, attn_dist, coverage_next = self.attention_d(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage, coverage)

        
         
        
        c_t_g, attn_dist_node,attn_dist_node_to_token,c_t_g2 = self.attention_f(s_t_hat,encoder_outputs_node, 
                                                           node_to_token, enc_padding_mask_node,
                                                           graph, flow)
        
        
        
        attn_dist_expand=attn_dist.unsqueeze(1).expand(b, nk, tk).contiguous()
        attn_dist_expand=attn_dist_expand * node_to_token
        flow_,indices = torch.max(attn_dist_expand, 2)
        normalization_factor = flow_.sum(1, keepdim=True)
        flow_next = flow_ / normalization_factor  
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_g, c_t_g2, s_t_hat), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        c_t_g = c_t_g * d_gra + c_t_g2 * (1-d_gra)       
        '''
        #print(d_gra)
        coverage = coverage_next

        flow = flow_next

        p_gen = None

        p_gen_input = torch.cat((c_t_d,c_t_g, c_t_g2, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)

        p_gen = self.p_gen_linear(p_gen_input)

        p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t_d, c_t_g, c_t_g2), 1) # B x hidden_dim * 3

        output = self.out1(output) # B x hidden_dim

        output = self.out2(output) # B x vocab_size

        vocab_dist = F.softmax(output, dim=1)
        
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_d, c_t_g, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        attn_dist = attn_dist * d_gra + attn_dist_node_to_token * (1-d_gra)
        '''

        nk=torch.sum(enc_padding_mask_node,1).unsqueeze(1)
        

        #attn_dist_all = attn_dist + attn_dist * attn_dist_node_to_token * nk 
        #attn_dist_all = attn_dist * (attn_dist_node_to_token + 0.1)
        attn_dist_all = attn_dist
        

       
        #print(step, i1==i2, torch.topk(attn_dist[0], 1).values,torch.topk(attn_dist_node[0], 1).values)
        #print('------------------')
        
        
        normalization_factor = attn_dist_all.sum(1, keepdim=True)
        attn_dist_all = attn_dist_all / normalization_factor


        
        #attn_dist_all = attn_dist * 0.6 + attn_dist_node_to_token * 0.4

        attn_dist_  = (1 - p_gen) * attn_dist_all

        vocab_dist_ = p_gen * vocab_dist

        final_dist = vocab_dist_.scatter_add(1, voc_d, attn_dist_)

        return final_dist, s_t, c_t_d, c_t_g,c_t_g2, attn_dist, attn_dist_node_to_token, p_gen, coverage,flow     


class LSTM_Decoder_new_x_share_emb(nn.Module):

    def __init__(self, config, emb_layer):

        super(LSTM_Decoder_new_x_share_emb, self).__init__()

        self.config=config

        self.attention_d = Doc_Attention(config)
        
        self.attention_f = Flow_Attention_only_node_improve_x(config)

        # decoder

        self.embedding = emb_layer
        #
        self.x_context = nn.Linear(config.hidden_dim * 4, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        #
        self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)

        self.d_gra_linear = nn.Linear(config.hidden_dim * 4, 1)
        
        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self.dropout = nn.Dropout(p=0.5)



    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_outputs_node, enc_padding_mask, enc_padding_mask_node,

                c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, graph, voc_d, step):

        b, tk, hid=encoder_outputs.size()
        b, nk, hid=encoder_outputs_node.size()

        y_t_1_embd = self.embedding(y_t_1)

        x = self.x_context(torch.cat((c_t_d,c_t_g, c_t_g2, y_t_1_embd), 1))
        
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t

        s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),

                             c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t_d, attn_dist, coverage_next = self.attention_d(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage, coverage)

        
         
        
        c_t_g, attn_dist_node,attn_dist_node_to_token,c_t_g2 = self.attention_f(s_t_hat,encoder_outputs_node, 
                                                           node_to_token, enc_padding_mask_node,
                                                           graph, flow)
        
        
        
        attn_dist_expand=attn_dist.unsqueeze(1).expand(b, nk, tk).contiguous()
        attn_dist_expand=attn_dist_expand * node_to_token
        flow_,indices = torch.max(attn_dist_expand, 2)
        normalization_factor = flow_.sum(1, keepdim=True)
        flow_next = flow_ / normalization_factor  
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_g, c_t_g2, s_t_hat), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        c_t_g = c_t_g * d_gra + c_t_g2 * (1-d_gra)       
        '''
        #print(d_gra)
        coverage = coverage_next

        flow = flow_next

        p_gen = None

        c_t_d = self.dropout(c_t_d)
        c_t_g = self.dropout(c_t_g)
        c_t_g2 = self.dropout(c_t_g2)

        p_gen_input = torch.cat((c_t_d,c_t_g, c_t_g2, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        
        #p_gen_input = self.dropout(p_gen_input)        

        p_gen = self.p_gen_linear(p_gen_input)

        p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t_d, c_t_g, c_t_g2), 1) # B x hidden_dim * 3

        #output = self.dropout(output)

        output = self.out1(output) # B x hidden_dim

        output = self.out2(output) # B x vocab_size

        vocab_dist = F.softmax(output, dim=1)
        
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_d, c_t_g, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        attn_dist = attn_dist * d_gra + attn_dist_node_to_token * (1-d_gra)
        '''

        nk=torch.sum(enc_padding_mask_node,1).unsqueeze(1)
        

        #attn_dist_all = attn_dist + attn_dist * attn_dist_node_to_token * nk 
        #attn_dist_all = attn_dist * (attn_dist_node_to_token + 0.1)
        attn_dist_all = attn_dist
        

       
        #print(step, i1==i2, torch.topk(attn_dist[0], 1).values,torch.topk(attn_dist_node[0], 1).values)
        #print('------------------')
        
        
        normalization_factor = attn_dist_all.sum(1, keepdim=True)
        attn_dist_all = attn_dist_all / normalization_factor


        
        #attn_dist_all = attn_dist * 0.6 + attn_dist_node_to_token * 0.4

        attn_dist_  = (1 - p_gen) * attn_dist_all

        vocab_dist_ = p_gen * vocab_dist

        final_dist = vocab_dist_.scatter_add(1, voc_d, attn_dist_)

        return final_dist, s_t, c_t_d, c_t_g,c_t_g2, attn_dist, attn_dist_node_to_token, p_gen, coverage,flow   





class LSTM_Decoder_new_x_share_emb_only_context(nn.Module):

    def __init__(self, config, emb_layer):

        super(LSTM_Decoder_new_x_share_emb_only_context, self).__init__()

        self.config=config

        self.attention_d = Doc_Attention(config)
        
        self.attention_f = Flow_Attention_only_node_improve_x(config)

        # decoder

        self.embedding = emb_layer
        #
        self.x_context = nn.Linear(config.hidden_dim * 3, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        #
        self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        self.d_gra_linear = nn.Linear(config.hidden_dim * 4, 1)
        
        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)

        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self.dropout = nn.Dropout(p=0.5)



    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_outputs_node, enc_padding_mask, enc_padding_mask_node,

                c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, graph, voc_d, step):

        b, tk, hid=encoder_outputs.size()
        b, nk, hid=encoder_outputs_node.size()

        y_t_1_embd = self.embedding(y_t_1)

        x = self.x_context(torch.cat((c_t_d,c_t_g, y_t_1_embd), 1))
        
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t

        s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),

                             c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t_d, attn_dist, coverage_next = self.attention_d(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage, coverage)

        
         
        
        c_t_g, attn_dist_node,attn_dist_node_to_token,c_t_g2 = self.attention_f(s_t_hat,encoder_outputs_node, 
                                                           node_to_token, enc_padding_mask_node,
                                                           graph, flow)
        
        
        
        attn_dist_expand=attn_dist.unsqueeze(1).expand(b, nk, tk).contiguous()
        attn_dist_expand=attn_dist_expand * node_to_token
        flow_,indices = torch.max(attn_dist_expand, 2)
        normalization_factor = flow_.sum(1, keepdim=True)
        flow_next = flow_ / normalization_factor  
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_g, c_t_g2, s_t_hat), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        c_t_g = c_t_g * d_gra + c_t_g2 * (1-d_gra)       
        '''
        #print(d_gra)
        coverage = coverage_next

        flow = flow_next

        p_gen = None

        c_t_d = self.dropout(c_t_d)
        c_t_g = self.dropout(c_t_g)
        c_t_g2 = self.dropout(c_t_g2)

        p_gen_input = torch.cat((c_t_d,c_t_g, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        
        #p_gen_input = self.dropout(p_gen_input)        

        p_gen = self.p_gen_linear(p_gen_input)

        p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t_d, c_t_g), 1) # B x hidden_dim * 3

        #output = self.dropout(output)

        output = self.out1(output) # B x hidden_dim

        output = self.out2(output) # B x vocab_size

        vocab_dist = F.softmax(output, dim=1)
        
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_d, c_t_g, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        attn_dist = attn_dist * d_gra + attn_dist_node_to_token * (1-d_gra)
        '''

        nk=torch.sum(enc_padding_mask_node,1).unsqueeze(1)
        

        #attn_dist_all = attn_dist + attn_dist * attn_dist_node_to_token * nk 
        #attn_dist_all = attn_dist * (attn_dist_node_to_token + 0.1)
        attn_dist_all = attn_dist
        

       
        #print(step, i1==i2, torch.topk(attn_dist[0], 1).values,torch.topk(attn_dist_node[0], 1).values)
        #print('------------------')
        
        
        normalization_factor = attn_dist_all.sum(1, keepdim=True)
        attn_dist_all = attn_dist_all / normalization_factor


        
        #attn_dist_all = attn_dist * 0.6 + attn_dist_node_to_token * 0.4

        attn_dist_  = (1 - p_gen) * attn_dist_all

        vocab_dist_ = p_gen * vocab_dist

        final_dist = vocab_dist_.scatter_add(1, voc_d, attn_dist_)

        return final_dist, s_t, c_t_d, c_t_g,c_t_g2, attn_dist, attn_dist_node_to_token, p_gen, coverage,flow   




class LSTM_Decoder_new_x_share_emb_only_flow(nn.Module):

    def __init__(self, config, emb_layer):

        super(LSTM_Decoder_new_x_share_emb_only_flow, self).__init__()

        self.config=config

        self.attention_d = Doc_Attention(config)
        
        self.attention_f = Flow_Attention_only_node_improve_x(config)

        # decoder

        self.embedding = emb_layer
        #
        self.x_context = nn.Linear(config.hidden_dim * 3, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        #
        self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        self.d_gra_linear = nn.Linear(config.hidden_dim * 4, 1)
        
        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)

        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self.dropout = nn.Dropout(p=0.5)



    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_outputs_node, enc_padding_mask, enc_padding_mask_node,

                c_t_d, c_t_g,c_t_g2, coverage, flow, node_to_token, graph, voc_d, step):

        b, tk, hid=encoder_outputs.size()
        b, nk, hid=encoder_outputs_node.size()

        y_t_1_embd = self.embedding(y_t_1)

        x = self.x_context(torch.cat((c_t_d,c_t_g2, y_t_1_embd), 1))
        
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t

        s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),

                             c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t_d, attn_dist, coverage_next = self.attention_d(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage, coverage)

        
         
        
        c_t_g, attn_dist_node,attn_dist_node_to_token,c_t_g2 = self.attention_f(s_t_hat,encoder_outputs_node, 
                                                           node_to_token, enc_padding_mask_node,
                                                           graph, flow)
        
        
        
        attn_dist_expand=attn_dist.unsqueeze(1).expand(b, nk, tk).contiguous()
        attn_dist_expand=attn_dist_expand * node_to_token
        flow_,indices = torch.max(attn_dist_expand, 2)
        normalization_factor = flow_.sum(1, keepdim=True)
        flow_next = flow_ / normalization_factor  
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_g, c_t_g2, s_t_hat), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        c_t_g = c_t_g * d_gra + c_t_g2 * (1-d_gra)       
        '''
        #print(d_gra)
        coverage = coverage_next

        flow = flow_next

        p_gen = None

        c_t_d = self.dropout(c_t_d)
        c_t_g = self.dropout(c_t_g)
        c_t_g2 = self.dropout(c_t_g2)

        p_gen_input = torch.cat((c_t_d,c_t_g2, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        
        #p_gen_input = self.dropout(p_gen_input)        

        p_gen = self.p_gen_linear(p_gen_input)

        p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t_d, c_t_g2), 1) # B x hidden_dim * 3

        #output = self.dropout(output)

        output = self.out1(output) # B x hidden_dim

        output = self.out2(output) # B x vocab_size

        vocab_dist = F.softmax(output, dim=1)
        
        '''
        d_gra = None       
        d_gra_input = torch.cat((c_t_d, c_t_g, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        d_gra = self.d_gra_linear(d_gra_input)
        d_gra = F.sigmoid(d_gra)
        attn_dist = attn_dist * d_gra + attn_dist_node_to_token * (1-d_gra)
        '''

        nk=torch.sum(enc_padding_mask_node,1).unsqueeze(1)
        

        #attn_dist_all = attn_dist + attn_dist * attn_dist_node_to_token * nk 
        #attn_dist_all = attn_dist * (attn_dist_node_to_token + 0.1)
        attn_dist_all = attn_dist
        

       
        #print(step, i1==i2, torch.topk(attn_dist[0], 1).values,torch.topk(attn_dist_node[0], 1).values)
        #print('------------------')
        
        
        normalization_factor = attn_dist_all.sum(1, keepdim=True)
        attn_dist_all = attn_dist_all / normalization_factor


        
        #attn_dist_all = attn_dist * 0.6 + attn_dist_node_to_token * 0.4

        attn_dist_  = (1 - p_gen) * attn_dist_all

        vocab_dist_ = p_gen * vocab_dist

        final_dist = vocab_dist_.scatter_add(1, voc_d, attn_dist_)

        return final_dist, s_t, c_t_d, c_t_g,c_t_g2, attn_dist, attn_dist_node_to_token, p_gen, coverage,flow   
