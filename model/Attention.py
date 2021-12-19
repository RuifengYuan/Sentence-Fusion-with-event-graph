# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:41:13 2020

@author: User
"""

import torch

import torch.nn as nn

import torch.nn.functional as F







class Attention(nn.Module):

    def __init__(self,config):

        super(Attention, self).__init__()

        # attention

        self.config=config

        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)
            
        self.encode_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v = nn.Linear(config.hidden_dim, 1, bias=False)



    def forward(self, s_t_hat, encoder_outputs, enc_padding_mask, coverage):

        b, t_k, n = list(encoder_outputs.size())
        
        encoder_feature = encoder_outputs.reshape(-1, self.config.hidden_dim)  # B * t_k x hidden_dim

        encoder_feature = self.encode_proj(encoder_feature)
        

        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim

        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x hidden_dim

        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x hidden_dim



        att_features = encoder_feature + dec_fea_expanded # B * t_k x hidden_dim



        coverage_input = coverage.view(-1, 1)  # B * t_k x 1

        coverage_feature = self.W_c(coverage_input)  # B * t_k x hidden_dim

        att_features = att_features + coverage_feature



        e = F.tanh(att_features) # B * t_k x hidden_dim

        scores = self.v(e)  # B * t_k x 1

        scores = scores.view(-1, t_k)  # B x t_k



        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k

        normalization_factor = attn_dist_.sum(1, keepdim=True)

        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k

        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n

        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim



        attn_dist = attn_dist.view(-1, t_k)  # B x t_k





        coverage = coverage.view(-1, t_k)

        coverage = coverage + attn_dist

        coverage = torch.clamp(coverage, 0, 1, out=None)

        return c_t, attn_dist, coverage
    
    
    
class Flow_Attention(nn.Module):

    def __init__(self,config):

        super(Flow_Attention, self).__init__()

        # attention

        self.config=config

        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)
            
        self.encode_proj_token = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.encode_proj_node = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v_token = nn.Linear(config.hidden_dim, 1, bias=False)

        self.v_node = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs_token, encoder_outputs_node, node_to_token, enc_padding_mask_token, enc_padding_mask_node, graph, flow):

        b, t_k, n = list(encoder_outputs_token.size())
        b, n_k, n = list(encoder_outputs_node.size())

        encoder_feature_token = encoder_outputs_token.view(-1, self.config.hidden_dim)  # B * t_k x hidden_dim

        encoder_feature_token = self.encode_proj_token(encoder_feature_token)
        
        encoder_feature_node = encoder_outputs_node.view(-1, self.config.hidden_dim)  # B * n_k x hidden_dim

        encoder_feature_node = self.encode_proj_node(encoder_feature_node)        
        
        

        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim

        dec_fea_expanded_token = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x hidden_dim

        dec_fea_expanded_token = dec_fea_expanded_token.view(-1, n)  # B * t_k x hidden_dim

        dec_fea_expanded_node = dec_fea.unsqueeze(1).expand(b, n_k, n).contiguous() # B x n_k x hidden_dim

        dec_fea_expanded_node = dec_fea_expanded_node.view(-1, n)  # B * n_k x hidden_dim
        

        #scores for nodes        
        flow = flow.unsqueeze(2)
        
        flow = flow.view(-1, 1)  # B * n_k x 1
        
        flow_feature = self.W_c(flow)  # B * n_k x hidden_dim        
        
        att_features_node = encoder_feature_node + dec_fea_expanded_node + flow_feature# B * n_k x hidden_dim
        
        e_node = F.tanh(att_features_node) # B * n_k x hidden_dim

        scores_node = self.v_node(e_node)  # B * n_k x 1

        scores_node = scores_node.view(-1, n_k)  # B x n_k

        attn_dist_node_ = F.softmax(scores_node, dim=1)*enc_padding_mask_node # B x n_k

        normalization_factor = attn_dist_node_.sum(1, keepdim=True)

        attn_dist_node = attn_dist_node_ / normalization_factor
        
        #node to token attention
        
        attn_dist_node_to_token_ = attn_dist_node.unsqueeze(2)  # B x n_k x 1
        
        attn_dist_node_to_token_ = attn_dist_node_to_token_[torch.arange(attn_dist_node_to_token_.size(0)).unsqueeze(1), node_to_token]
        
        attn_dist_node_to_token_ = attn_dist_node_to_token_.view(-1, t_k)   # B x t_k 
        
        attn_dist_node_to_token_ = F.softmax(attn_dist_node_to_token_, dim=1)*enc_padding_mask_token # B x n_k

        normalization_factor = attn_dist_node_to_token_.sum(1, keepdim=True)

        attn_dist_node_to_token = attn_dist_node_to_token_ / normalization_factor  
        
        
        
        #scores for tokens
        
        att_features_token = encoder_feature_token + dec_fea_expanded_token # B * t_k x hidden_dim
        
        e_token = F.tanh(att_features_token) # B * t_k x hidden_dim

        scores_token = self.v_token(e_token)  # B * t_k x 1

        scores_token = scores_token.view(-1, t_k)  # B x t_k
        
        attn_dist_token_ = F.softmax(scores_token, dim=1)*enc_padding_mask_token # B x t_k

        normalization_factor = attn_dist_token_.sum(1, keepdim=True)

        attn_dist_token = attn_dist_token_ / normalization_factor        


        final_attn= 0.5 * attn_dist_token + 0.5 * attn_dist_node_to_token


       

        final_attn = final_attn.unsqueeze(1)  # B x 1 x t_k

        c_t = torch.bmm(final_attn, encoder_outputs_token)  # B x 1 x n

        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim



        final_attn = final_attn.view(-1, t_k)  # B x t_k

        zero_hop=attn_dist_node.unsqueeze(1)
  
        one_hop=torch.matmul(zero_hop, graph)
        
        two_hop= torch.matmul(one_hop, graph)
        
        flow_vector = zero_hop*0.33333 + one_hop*0.33333 + two_hop*0.33333   #B X nk
 
        flow = flow_vector.view(-1, n_k)   

        return c_t, final_attn, attn_dist_node, flow
    
    
class Flow_Attention_only_node(nn.Module):

    def __init__(self,config):

        super(Flow_Attention_only_node, self).__init__()

        # attention

        self.config=config

        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)

        self.encode_proj_node = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v_node = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t_hat,encoder_outputs_node, node_to_token,enc_padding_mask_node, graph, flow):

                

        b, n_k, t_k = list(node_to_token.size())
        
        b, n_k, n = list(encoder_outputs_node.size())
        
        encoder_feature_node = encoder_outputs_node.view(-1, self.config.hidden_dim)  # B * n_k x hidden_dim

        encoder_feature_node = self.encode_proj_node(encoder_feature_node)        
        
        
        
        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim

        dec_fea_expanded_node = dec_fea.unsqueeze(1).expand(b, n_k, n).contiguous() # B x n_k x hidden_dim

        dec_fea_expanded_node = dec_fea_expanded_node.view(-1, n)  # B * n_k x hidden_dim
        

        #scores for nodes        
        flow = flow.unsqueeze(2)
        
        flow = flow.view(-1, 1)  # B * n_k x 1
        
        flow_feature = self.W_c(flow)  # B * n_k x hidden_dim        
        
        #att_features_node = encoder_feature_node + dec_fea_expanded_node + flow_feature# B * n_k x hidden_dim
        att_features_node = encoder_feature_node + dec_fea_expanded_node
        e_node = F.tanh(att_features_node) # B * n_k x hidden_dim

        scores_node = self.v_node(e_node)  # B * n_k x 1

        scores_node = scores_node.view(-1, n_k)  # B x n_k

        attn_dist_node_ = F.softmax(scores_node, dim=1)*enc_padding_mask_node # B x n_k

        normalization_factor = attn_dist_node_.sum(1, keepdim=True)

        attn_dist_node = attn_dist_node_ / normalization_factor
        
        attn_dist_nodex = attn_dist_node.unsqueeze(1)  # B x 1 x n_k
        
        attn_dist_node_to_token_ = torch.bmm(attn_dist_nodex, node_to_token)
        
        attn_dist_node_to_token_=attn_dist_node_to_token_.view(-1, t_k)
        
        normalization_factor = attn_dist_node_to_token_.sum(1, keepdim=True)

        attn_dist_node_to_token = attn_dist_node_to_token_ / normalization_factor        
        
        
        c_t = torch.bmm(attn_dist_nodex, encoder_outputs_node)  # B x 1 x n

        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim



        zero_hop=attn_dist_node.unsqueeze(1)
  
        one_hop=torch.matmul(zero_hop, graph)
        
        two_hop= torch.matmul(one_hop, graph)
        
        flow_vector = zero_hop*0.33333 + one_hop*0.33333 + two_hop*0.33333   #B X nk
 
        flow = flow_vector.view(-1, n_k)   

        return c_t, attn_dist_node, attn_dist_node_to_token, flow    
    
    
    
class Flow_Attention_only_node_improve(nn.Module):

    def __init__(self,config):

        super(Flow_Attention_only_node_improve, self).__init__()

        # attention

        self.config=config

        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)

        self.encode_proj_node = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v_node = nn.Linear(config.hidden_dim, 1, bias=False)
        
        self.c_project = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.s_project = nn.Linear(config.hidden_dim*2, config.hidden_dim, bias=False)
        
        self.flow_trans = nn.Linear(config.hidden_dim, 3, bias=False)

    def forward(self, s_t_hat, encoder_outputs_node, node_to_token,enc_padding_mask_node, graph, flow):
        b, n_k, t_k = list(node_to_token.size())
        b, n_k, n = list(encoder_outputs_node.size())
        encoder_feature_node = encoder_outputs_node.view(-1, self.config.hidden_dim)  # B * n_k x hidden_dim
        encoder_feature_node = self.encode_proj_node(encoder_feature_node)        
        
                
        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim
        dec_fea_expanded_node = dec_fea.unsqueeze(1).expand(b, n_k, n).contiguous() # B x n_k x hidden_dim
        dec_fea_expanded_node = dec_fea_expanded_node.view(-1, n)  # B * n_k x hidden_dim
        

        #scores for nodes             
        att_features_node = encoder_feature_node + dec_fea_expanded_node# B * n_k x hidden_dim
        e_node = F.tanh(att_features_node) # B * n_k x hidden_dim
        scores_node = self.v_node(e_node)  # B * n_k x 1
        scores_node = scores_node.view(-1, n_k)  # B x n_k
        attn_dist_node_ = F.softmax(scores_node, dim=1)*enc_padding_mask_node # B x n_k
        
        #topk, indices = torch.topk(attn_dist_node_, 5)
        #res = torch.zeros(b, n_k, dtype=torch.float).cuda()
        #attn_dist_node_ = res.scatter(1, indices, topk)
        
        normalization_factor = attn_dist_node_.sum(1, keepdim=True)
        attn_dist_node = attn_dist_node_ / normalization_factor
        
        
        #scores for flow

        zero_hop=flow.unsqueeze(1)
        
        one_hop=torch.bmm(zero_hop, graph)
        one_hop=one_hop.view(-1, n_k)  
        normalization_factor = one_hop.sum(1, keepdim=True)
        one_hop = one_hop / (normalization_factor + 1e-10)   
        one_hop=one_hop.unsqueeze(1)   
        
        two_hop= torch.bmm(one_hop, graph) #B X 1 X nk
        two_hop=two_hop.view(-1, n_k)  
        normalization_factor = two_hop.sum(1, keepdim=True)
        two_hop = two_hop / (normalization_factor + 1e-10)        

        
        c_t_last = torch.bmm(flow.unsqueeze(1), encoder_outputs_node)    #B X 1 X hidden_dim
        c_fea = self.c_project(c_t_last)  #B X 1 X hidden_dim
        s_fea = self.s_project(s_t_hat)  #B X 1 X hidden_dim
        s_fea = s_fea.unsqueeze(1)
        f_fea = F.tanh(c_fea + s_fea)
        flow_gate = self.flow_trans(f_fea)  #B X 1 X 3
        flow_gate = F.softmax(flow_gate, dim=2)

        flow_gate =flow_gate.expand(b, n_k, 3).contiguous()
        
        zero_hop=zero_hop.view(-1, n_k) 
        #print(torch.topk(zero_hop[0], 3))
        zero_hop=zero_hop.unsqueeze(2) 
        
        
        one_hop=one_hop.view(-1, n_k)   
        #print(torch.topk(one_hop[0], 3))
        one_hop=one_hop.unsqueeze(2) 
        
        
        two_hop=two_hop.view(-1, n_k)
        #print(torch.topk(two_hop[0], 3))
        two_hop=two_hop.unsqueeze(2) 
        
        
        all_hop=torch.cat((zero_hop,one_hop,two_hop), 2) #b x nk x 3
        all_hop = all_hop * flow_gate  #b x nk x 3
        flow_attention_score = torch.sum(all_hop, 2)
        flow_attention_score.view(-1, n_k)  
        attn_dist_node = attn_dist_node * 1 + flow_attention_score * 0
        
        '''
        topk, indices = torch.topk(attn_dist_node, 5)
        res = torch.zeros(b, n_k, dtype=torch.float).cuda()
        attn_dist_node = res.scatter(1, indices, topk)
        
        normalization_factor = attn_dist_node.sum(1, keepdim=True)
        attn_dist_node = attn_dist_node / normalization_factor
        print(attn_dist_node)
        '''
        
        attn_dist_nodex = attn_dist_node.unsqueeze(1)  # B x 1 x n_k
        attn_dist_node_to_token_ = torch.bmm(attn_dist_nodex, node_to_token)
        attn_dist_node_to_token_=attn_dist_node_to_token_.view(-1, t_k)
        
        #normalization_factor = attn_dist_node_to_token_.sum(1, keepdim=True)
        #attn_dist_node_to_token = attn_dist_node_to_token_ / normalization_factor  
        attn_dist_node_to_token = attn_dist_node_to_token_
        
        c_t = torch.bmm(attn_dist_nodex, encoder_outputs_node)  # B x 1 x n
        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim

        return c_t, attn_dist_node, attn_dist_node_to_token, attn_dist_node  


class Flow_Attention_only_node_improve_x(nn.Module):

    def __init__(self,config):

        super(Flow_Attention_only_node_improve_x, self).__init__()

        # attention

        self.config=config

        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)

        self.encode_proj_node = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v_node = nn.Linear(config.hidden_dim, 1, bias=False)
        
        self.c_project = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.s_project = nn.Linear(config.hidden_dim*2, config.hidden_dim, bias=False)
        
        self.flow_trans = nn.Linear(config.hidden_dim, 3, bias=False)

    def forward(self, s_t_hat, encoder_outputs_node, node_to_token,enc_padding_mask_node, graph, flow):
        b, n_k, t_k = list(node_to_token.size())
        b, n_k, n = list(encoder_outputs_node.size())
        encoder_feature_node = encoder_outputs_node.view(-1, self.config.hidden_dim)  # B * n_k x hidden_dim
        encoder_feature_node = self.encode_proj_node(encoder_feature_node)        
        
                
        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim
        dec_fea_expanded_node = dec_fea.unsqueeze(1).expand(b, n_k, n).contiguous() # B x n_k x hidden_dim
        dec_fea_expanded_node = dec_fea_expanded_node.view(-1, n)  # B * n_k x hidden_dim
        

        #scores for nodes             
        att_features_node = encoder_feature_node + dec_fea_expanded_node# B * n_k x hidden_dim
        e_node = F.tanh(att_features_node) # B * n_k x hidden_dim
        scores_node = self.v_node(e_node)  # B * n_k x 1
        scores_node = scores_node.view(-1, n_k)  # B x n_k
        attn_dist_node_ = F.softmax(scores_node, dim=1)*enc_padding_mask_node # B x n_k
        
        #topk, indices = torch.topk(attn_dist_node_, 5)
        #res = torch.zeros(b, n_k, dtype=torch.float).cuda()
        #attn_dist_node_ = res.scatter(1, indices, topk)
        
        normalization_factor = attn_dist_node_.sum(1, keepdim=True)
        attn_dist_node = attn_dist_node_ / normalization_factor
        
        
        #scores for flow
        zero_hop=flow.unsqueeze(1)
        
        one_hop=torch.bmm(zero_hop, graph)
        one_hop=one_hop.view(-1, n_k)  
        normalization_factor = one_hop.sum(1, keepdim=True)
        one_hop = one_hop / (normalization_factor + 1e-10)   
        one_hop=one_hop.unsqueeze(1)   
        
        two_hop= torch.bmm(one_hop, graph) #B X 1 X nk
        two_hop=two_hop.view(-1, n_k)  
        normalization_factor = two_hop.sum(1, keepdim=True)
        two_hop = two_hop / (normalization_factor + 1e-10)        

        
        c_t_last = torch.bmm(flow.unsqueeze(1), encoder_outputs_node)    #B X 1 X hidden_dim
        c_fea = self.c_project(c_t_last)  #B X 1 X hidden_dim
        s_fea = self.s_project(s_t_hat)  #B X 1 X hidden_dim
        s_fea = s_fea.unsqueeze(1)
        f_fea = F.tanh(c_fea + s_fea)
        flow_gate = self.flow_trans(f_fea)  #B X 1 X 3
        flow_gate = F.softmax(flow_gate, dim=2)
        #print('-------------------------------')
        #print(flow_gate)
        #flow_gate = torch.tensor([[[0.5,0.5-0.0000001,0.0000001]]])
        #flow_gate = flow_gate.cuda()
        
        flow_gate =flow_gate.expand(b, n_k, 3).contiguous()
        
        zero_hop=zero_hop.view(-1, n_k) 
        #print(torch.topk(zero_hop[0], 3))
        zero_hop=zero_hop.unsqueeze(2)
        
        
        one_hop=one_hop.view(-1, n_k)   
        #print(torch.topk(one_hop[0], 3))
        one_hop=one_hop.unsqueeze(2)
        
        
        two_hop=two_hop.view(-1, n_k)
        #print(torch.topk(two_hop[0], 3))
        two_hop=two_hop.unsqueeze(2)
        
        
        all_hop=torch.cat((zero_hop,one_hop,two_hop), 2) #b x nk x 3
      
        all_hop = all_hop * flow_gate  #b x nk x 3
        flow_attention_score = torch.sum(all_hop, 2)

        
        #flow_attention_score = zero_hop.view(-1, n_k)*0.33333 + one_hop.view(-1, n_k)*0.33333 + two_hop.view(-1, n_k)*0.33333
        
        flow_attention_score.view(-1, n_k)  

        #print('flow_attention_score',torch.topk(flow_attention_score[0], 1).indices)
        #print('flow_attention_score',torch.topk(flow_attention_score[0], 1).values)
        #print('attn_dist_node',torch.topk(attn_dist_node[0], 1).indices)
        #print('attn_dist_node',torch.topk(attn_dist_node[0], 1).values)
        attn_dist_node_all = attn_dist_node * 0 + flow_attention_score * 1
        #print(torch.topk(attn_dist_node_all[0], 1).indices)        
        #print('--------------------------------------')
        '''
        topk, indices = torch.topk(attn_dist_node, 5)
        res = torch.zeros(b, n_k, dtype=torch.float).cuda()
        attn_dist_node = res.scatter(1, indices, topk)
        
        normalization_factor = attn_dist_node.sum(1, keepdim=True)
        attn_dist_node = attn_dist_node / normalization_factor
        print(attn_dist_node)
        '''
        
        attn_dist_nodex = attn_dist_node_all.unsqueeze(1)  # B x 1 x n_k
        attn_dist_node_to_token_ = torch.bmm(attn_dist_nodex, node_to_token)
        attn_dist_node_to_token_=attn_dist_node_to_token_.view(-1, t_k)
        
        #normalization_factor = attn_dist_node_to_token_.sum(1, keepdim=True)
        #attn_dist_node_to_token = attn_dist_node_to_token_ / normalization_factor  
        attn_dist_node_to_token = attn_dist_node_to_token_
        
        c_t = torch.bmm(attn_dist_node.unsqueeze(1), encoder_outputs_node)  # B x 1 x n
        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim

        c_t2 = torch.bmm(flow_attention_score.unsqueeze(1), encoder_outputs_node)  # B x 1 x n
        c_t2 = c_t2.view(-1, self.config.hidden_dim)  # B x hidden_dim

        return c_t, attn_dist_node_all, (attn_dist_node,flow_attention_score,scores_node), c_t2



class Doc_Attention(nn.Module):

    def __init__(self,config):

        super(Doc_Attention, self).__init__()

        # attention

        self.config=config

        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)
            
        self.encode_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v = nn.Linear(config.hidden_dim, 1, bias=False)



    def forward(self, s_t_hat, encoder_outputs, enc_padding_mask, coverage, attn_dist_node_to_token):

        b, t_k, n = list(encoder_outputs.size())
        

        encoder_feature = encoder_outputs.view(-1, self.config.hidden_dim)  # B * t_k x hidden_dim

        encoder_feature = self.encode_proj(encoder_feature)
        

        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim

        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x hidden_dim

        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x hidden_dim



        att_features = encoder_feature + dec_fea_expanded # B * t_k x hidden_dim



        coverage_input = coverage.view(-1, 1)  # B * t_k x 1

        coverage_feature = self.W_c(coverage_input)  # B * t_k x hidden_dim

        att_features = att_features + coverage_feature



        e = F.tanh(att_features) # B * t_k x hidden_dim

        scores = self.v(e)  # B * t_k x 1

        scores = scores.view(-1, t_k)  # B x t_k



        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k

        normalization_factor = attn_dist_.sum(1, keepdim=True)

        attn_dist = attn_dist_ / normalization_factor
        
        
#        final_attn = attn_dist * 0.6 + attn_dist_node_to_token * 0.4
        final_attn = attn_dist * 1        
        
        

        final_attn = final_attn.unsqueeze(1)  # B x 1 x t_k

        c_t = torch.bmm(final_attn, encoder_outputs)  # B x 1 x n

        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim



        final_attn = final_attn.view(-1, t_k)  # B x t_k



        coverage = coverage.view(-1, t_k)

        coverage = coverage + attn_dist

        coverage = torch.clamp(coverage, 0, 1, out=None)

        return c_t, final_attn, coverage






class Flow_Attention_with_Coverage(nn.Module):

    def __init__(self,config):

        super(Flow_Attention_with_Coverage, self).__init__()

        # attention

        self.config=config

        self.W_f = nn.Linear(1, config.hidden_dim, bias=False)
        
        self.W_c = nn.Linear(1, config.hidden_dim, bias=False)
            
        self.encode_proj_token = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.encode_proj_node = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim)

        self.v_token = nn.Linear(config.hidden_dim, 1, bias=False)

        self.v_node = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs_token, encoder_outputs_node, node_to_token, enc_padding_mask_token, enc_padding_mask_node, graph, flow, coverage):

        b, t_k, n = list(encoder_outputs_token.size())
        b, n_k, n = list(encoder_outputs_node.size())

        encoder_feature_token = encoder_outputs_token.view(-1, self.config.hidden_dim)  # B * t_k x hidden_dim

        encoder_feature_token = self.encode_proj_token(encoder_feature_token)
        
        encoder_feature_node = encoder_outputs_node.view(-1, self.config.hidden_dim)  # B * n_k x hidden_dim

        encoder_feature_node = self.encode_proj_node(encoder_feature_node)        
        
        

        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim

        dec_fea_expanded_token = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x hidden_dim

        dec_fea_expanded_token = dec_fea_expanded_token.view(-1, n)  # B * t_k x hidden_dim

        dec_fea_expanded_node = dec_fea.unsqueeze(1).expand(b, n_k, n).contiguous() # B x n_k x hidden_dim

        dec_fea_expanded_node = dec_fea_expanded_node.view(-1, n)  # B * n_k x hidden_dim
        

        #scores for nodes    
        flow = flow.unsqueeze(2)
        
        flow = flow.view(-1, 1)  # B * n_k x 1

        flow_feature = self.W_f(flow)  # B * t_k x hidden_dim 
        
        att_features_node = encoder_feature_node + dec_fea_expanded_node + flow_feature # B * t_k x hidden_dim
        
        e_node = F.tanh(att_features_node) # B * n_k x hidden_dim

        scores_node = self.v_node(e_node)  # B * n_k x 1

        scores_node = scores_node.view(-1, n_k)  # B x n_k

        attn_dist_node_ = F.softmax(scores_node, dim=1)*enc_padding_mask_node # B x n_k

        normalization_factor = attn_dist_node_.sum(1, keepdim=True)

        attn_dist_node = attn_dist_node_ / normalization_factor
        

        
        #node to token attention
        
        attn_dist_node_to_token_ = attn_dist_node.unsqueeze(2)  # B x n_k x 1
        
        attn_dist_node_to_token_ = attn_dist_node_to_token_[torch.arange(attn_dist_node_to_token_.size(0)).unsqueeze(1), node_to_token]
        
        attn_dist_node_to_token_ = attn_dist_node_to_token_.view(-1, t_k)   # B x t_k 
        
        attn_dist_node_to_token_ = F.softmax(attn_dist_node_to_token_, dim=1)*enc_padding_mask_token # B x n_k

        normalization_factor = attn_dist_node_to_token_.sum(1, keepdim=True)

        attn_dist_node_to_token = attn_dist_node_to_token_ / normalization_factor  
        
        
        
        #scores for tokens
        coverage_input = coverage.view(-1, 1)  # B * t_k x 1

        coverage_feature = self.W_c(coverage_input)  # B * t_k x hidden_dim

        att_features_token = encoder_feature_token + dec_fea_expanded_token + coverage_feature# B * t_k x hidden_dim
        
        e_token = F.tanh(att_features_token) # B * t_k x hidden_dim

        scores_token = self.v_token(e_token)  # B * t_k x 1

        scores_token = scores_token.view(-1, t_k)  # B x t_k
        
        attn_dist_token_ = F.softmax(scores_token, dim=1)*enc_padding_mask_token # B x t_k

        normalization_factor = attn_dist_token_.sum(1, keepdim=True)

        attn_dist_token = attn_dist_token_ / normalization_factor        

        final_attn= 0.5 * attn_dist_token + 0.5 * attn_dist_node_to_token


       

        final_attn = final_attn.unsqueeze(1)  # B x 1 x t_k

        c_t = torch.bmm(final_attn, encoder_outputs_token)  # B x 1 x n

        c_t = c_t.view(-1, self.config.hidden_dim)  # B x hidden_dim



        final_attn = final_attn.view(-1, t_k)  # B x t_k

        coverage = coverage.view(-1, t_k)

        coverage = coverage + final_attn
        
        coverage = torch.clamp(coverage, 0, 1, out=None)
        
        

        zero_hop=attn_dist_node.unsqueeze(1)
  
        one_hop=torch.matmul(zero_hop, graph)
        
        two_hop= torch.matmul(one_hop, graph)
        
        flow_vector = zero_hop*0.33333 + one_hop*0.33333 + two_hop*0.33333   #B X nk
 
        flow = flow_vector.view(-1, n_k)       


        return c_t, final_attn, attn_dist_node, coverage, flow