import torch
import numpy as np
import pickle
import copy
import time
import glob

stop_word=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
           "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
           "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
           "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
           "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
           "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
           "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
           "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
           "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
           "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
           "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
           "now","s","t","'","(",")",",",'"','us']


def pad_with_mask(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    
    pad_mask = [[1] * len(d) + [0] * (width - len(d)) for d in data]
    return rtn_data,pad_mask

def pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]

    return rtn_data

def h_mask(graph, src, clss, cls_id, seq_id, pad_id):
    mask=np.zeros((len(src), len(src)), dtype=np.float16)

    
    for i in range (len(src)):
        mask[i][i]=1
            
    for i,tokeni in enumerate(src):
        if tokeni == pad_id:
            break
        elif tokeni == cls_id:
            start=i+1
            end=0
            for k in range(i+1,i+100):
                if k == len(src)-1:
                    end=copy.deepcopy(k)
                    mask[i][k]=1
                    mask[k][i]=1                   
                    break
                if src[k]==cls_id:
                    break
                end=copy.deepcopy(k)
                mask[i][k]=1
                mask[k][i]=1
            for kk in range(start,end+1):
                mask[kk][start:end+1]=1
        else:
            pass
    
    for edge in graph.get_edgelist():
        n1=edge[0]
        n2=edge[1]
        p1=clss[n1]
        p2=clss[n2]
        mask[p1][p2]=1
        mask[p2][p1]=1
    return mask



def data_loader_train_split_new(article_no_res,abstract,graph,tokenizer,arg):
    
    cls_id=101
    seq_id=102
    mask_id=103
    end_id = 100
    pad_id=0
    dec_start_id=1
    
    #encoder for document  
    #prepear src_d  pos_d 
    pre_src_d=[]
    pre_pos_d=[]
    doc_len=[]
    
    sent_to_pos_list=[]
    
    for one in article_no_res:

        #first split for this example

        one_pre_src_d=[]
        one_pre_pos_d=[]
        one_doc_len=[]
        sent_to_pos={}
        d_id=0
        bias=0

        count=0
        s_id=0
        for sent in one:
            sent_encode=tokenizer.encode(sent,add_special_tokens=False)
            sent_encode=sent_encode+[seq_id]
            len_sent=len(sent_encode)
            if len_sent<arg.min_sent_len:
                s_id=s_id+1
                continue
            if count+len_sent > arg.max_len:
                break
            else:
                sent_to_pos[str(d_id)+'-'+str(s_id)]=(count+bias,count+len_sent+bias)
                count=count+len_sent
                one_pre_src_d=one_pre_src_d+sent_encode
            s_id=s_id+1

        bias=bias+count
        one_doc_len.append(count)
        one_pre_pos_d=one_pre_pos_d+list(range(count))
            
        pre_src_d.append(one_pre_src_d)
        pre_pos_d.append(one_pre_pos_d)
        doc_len.append(one_doc_len)
        
        sent_to_pos_list.append(sent_to_pos)
    
    
    src_d, pad_mask_d=torch.tensor(pad_with_mask(pre_src_d, pad_id))  
    pos_d, pad_mask_d=torch.tensor(pad_with_mask(pre_pos_d, pad_id)) 
                
    #prepear attention mask
    mask_d=[]
    for one in doc_len:
        one_mask_d=np.zeros((len(src_d[0]), len(src_d[0])), dtype=np.float16)
        count=0
        for i in one:
            for x in range(count,count+i):
                one_mask_d[x][count:count+i]=1
            count=count+i
        mask_d.append(one_mask_d)
        
    mask_d=torch.tensor(mask_d, dtype=torch.float) 
    
    #encoder for graph
    #filter the graph

    filter_graph=[]
    for one_id,one in enumerate(graph):
        del_node=[]

        for node_id,node_content in enumerate(one.vs["position"]):
            for j in node_content:
                d=0
                s=j
                k=str(d)+'-'+str(s)            
                if k not in sent_to_pos_list[one_id].keys():
                    if k not in del_node:
                        del_node.append(node_id)

        
        new_one = copy.deepcopy(one)
        new_one.delete_vertices(del_node)
        if arg.add_root == 1:
            pass
        
        filter_graph.append(new_one)

    
    #prepear 
    src_d_merge = src_d
    max_node=max([len(one.vs['content']) for one in filter_graph])
    
    
    pre_token_to_node=[]
    pre_node_to_token=[]
    pre_node_mask=[]
    merge_tk=len(src_d_merge[0])
    for one_id,one in enumerate(filter_graph):
        #print(sent_to_pos_list[one_id])
        #print(one.vs['position'])
        one_token_to_node=[]
        one_node_to_token=[]
        one_node_mask=[]
        for node_id,node in enumerate(one.vs['content']):
            node_encode=tokenizer.encode(node,add_special_tokens=False)
            node_encode_str=[str(i) for i in node_encode]
            node_encode_str=' '.join(node_encode_str)
            mask_encode=len(node_encode)*['-1']
            mask_encode_str=' '.join(mask_encode)
            sent_pos=one.vs['position'][node_id]
            node_pos=[0]*merge_tk
            for j in sent_pos:
                d=0
                s=j
                k=str(d)+'-'+str(s)
                rangex=sent_to_pos_list[one_id][k]
                tar_sent=src_d_merge[one_id][rangex[0]:rangex[1]]
                tar_sent_str=[str(int(i)) for i in tar_sent]
                tar_sent_str=' '.join(tar_sent_str)
                tar_sent_str=tar_sent_str.replace(node_encode_str,mask_encode_str)
                tar_sent = tar_sent_str.split()
                for p,x in enumerate(tar_sent):
                    if x == '-1':
                        node_pos[p+rangex[0]]=1
            '''
            print(one.vs['content'][node_id])
            print(one.vs['position'][node_id])
            xx=[ii for ii,i in enumerate(node_pos) if i != 0]
            print(xx)
            xxx=[src_d_merge[one_id][i] for i in xx]
            print(tokenizer.decode(xxx))
            print('----------------------')
            '''
            one_node_to_token.append(node_pos)
            z=sum(node_pos)
            if z != 0:
                node_pos=[i/z for i in node_pos]
                one_token_to_node.append(node_pos)
                one_node_mask.append(1)
            else:
                one_token_to_node.append(node_pos)
                one_node_mask.append(0)                

        for i in range(max_node-len(one_node_mask)):
            one_token_to_node.append([0]*merge_tk)
            one_node_to_token.append([0]*merge_tk)
            one_node_mask.append(0)
        
        pre_token_to_node.append(one_token_to_node)
        pre_node_to_token.append(one_node_to_token)
        pre_node_mask.append(one_node_mask)  
        
    token_to_node=torch.tensor(pre_token_to_node,dtype=torch.float)
    node_to_token=torch.tensor(pre_node_to_token,dtype=torch.float)
    node_mask=torch.tensor(pre_node_mask,dtype=torch.float)    
    
    #prepear adj_graph
    adj_graph=[]
    for i,one in enumerate(filter_graph):
        one_adj=np.zeros((max_node, max_node), dtype=np.float16)
        for edge in one.get_edgelist():
            n1=edge[0]
            n2=edge[1]

            one_adj[n1][n2]=1
            one_adj[n2][n1]=1
            one_adj[n1][n1]=1  
            one_adj[n2][n2]=1  
            
        for line in range(max_node):
            sum_line=np.sum(one_adj[line])
            if sum_line > 0:
                one_adj[line]=one_adj[line]/sum_line
        adj_graph.append(one_adj)
        
    adj_graph=torch.tensor(adj_graph, dtype=torch.float)
    
    #prepear flow_graph
    flow_graph=[]
    for i,one in enumerate(filter_graph):
        one_flow=np.zeros((max_node, max_node), dtype=np.float16)
        for edge in one.get_edgelist():
            n1=edge[0]
            n2=edge[1]
            if n1 != n2:
                #if one_flow[n1][n2] == 0 and one_flow[n2][n1] == 0:
                one_flow[n1][n2]=1
                #one_flow[n2][n1]=1*arg.one_direction_reduction 
             
        for line in range(max_node):
            sum_line=np.sum(one_flow[line])
            if sum_line > 0:
                one_flow[line]=one_flow[line]/sum_line
        flow_graph.append(one_flow)
        
    flow_graph=torch.tensor(flow_graph, dtype=torch.float)    
    
    
    #Decode
    pre_inp=[]
    pre_tar=[]
    dec_len=[]
    for one in abstract:
        one_dec_encode=[]
        
        dec_encode=tokenizer.encode(one,add_special_tokens=False)
        one_dec_encode=one_dec_encode+dec_encode
        one_dec_encode=one_dec_encode+[end_id]

        one_pre_inp=[seq_id]+one_dec_encode
        one_pre_tar=one_dec_encode
        dec_len.append(min(len(one_dec_encode),arg.max_dec_steps))
        pre_inp.append(one_pre_inp)
        pre_tar.append(one_pre_tar)
    

    inp=torch.tensor(pad(pre_inp, 0)) 
    
    tar,tar_mask=torch.tensor(pad_with_mask(pre_tar, 0))  
    
    max_dec_len=max(dec_len)
    max_dec_len=torch.tensor(max_dec_len)
    dec_len=torch.tensor(dec_len)   



    pre_oracle_label=[]
    for i,one in enumerate(filter_graph):    
        one_oracle_label=one.vs['cluster']

        pre_oracle_label.append(one_oracle_label)
        
    oracle_label= torch.tensor(pad(pre_oracle_label, pad_id)) 

    pos_embedding=[]
    '''
    for one in node_to_token:
        tk=len(one[0])
        one_pos_embedding=[]
        for n in one:
            if tk >= arg.pos_dim:
                one_pos_embedding.append(n[:arg.pos_dim-1])
            else:
                one_pos_embedding.append(list(n)+[0]*(arg.pos_dim-tk))               
        pos_embedding.append(one_pos_embedding)    
    
    pos_embedding=torch.tensor(pos_embedding)
    '''

    '''
    print(abs_key[0])
    print(filter_graph[0].vs['content'])
    print(oracle_label[0])
    
    gg=copy.deepcopy(filter_graph[0])
    gg.vs['cluster'] = [int(i) for i in oracle_label[0]]
    igraph2graphviz(gg)
    '''
    #igraph2graphviz(filter_graph[0])    
    '''
    torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
    print('token_to_node:')
    print(token_to_node)
    print('node_to_token:')
    print(node_to_token)
    print(node_to_token.size())
    print('node_mask')
    print(node_mask)
    print('adj_graph')
    print(adj_graph)
    print('flow_graph')
    print(flow_graph)
    '''

    '''
    for i,one in enumerate(node_to_token[0]):
        
        xxx=filter_graph[0].vs['content'][i]
        print(xxx)   
        print(tokenizer.encode(xxx,add_special_tokens=False))
        print(one * src_d_merge[0])
    '''
    '''
    print('inp')
    print(inp)      
    print('tar')
    print(tar)      
    print('tar_mask')
    print(tar_mask)      
    print('dec_len')
    print(dec_len)      
    print('max_dec_len')
    print(max_dec_len)  
    print('doc_len')
    print(len(src_d[0])) 
    print('--------------------------')
    '''

    src_d=src_d.cuda()
    pos_d=pos_d.cuda()
    mask_d=mask_d.cuda()
    pad_mask_d=pad_mask_d.cuda()
    
    token_to_node=token_to_node.cuda()
    node_to_token=node_to_token.cuda()
    node_mask=node_mask.cuda()
    adj_graph=adj_graph.cuda()
    flow_graph=flow_graph.cuda()
    
    inp=inp.cuda()
    tar=tar.cuda()
    tar_mask=tar_mask.cuda()
    dec_len=dec_len.cuda()
    max_dec_len=max_dec_len.cuda()

    #pos_embedding=pos_embedding.cuda()
    return src_d,pos_d,mask_d,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding








def data_loader_train_bert(article_no_res,abstract,graph,tokenizer,arg):
    
    cls_id=101
    seq_id=102
    mask_id=103
    end_id = 100
    pad_id=0
    dec_start_id=1
    
    #encoder for document  
    #prepear src_d  pos_d 
    pre_src_d=[]
    pre_pos_d=[]
    doc_len=[]
    
    sent_to_pos_list=[]
    


    
    
    for one in article_no_res:

        #first split for this example

        one_pre_src_d=[]
        one_pre_pos_d=[]
        one_doc_len=[]
        sent_to_pos={}
        d_id=0
        bias=0

        count=0
        s_id=0
        for sent in one:
            sent_encode=tokenizer.encode(sent,add_special_tokens=False)
            sent_encode=sent_encode+[seq_id]
            len_sent=len(sent_encode)
            if len_sent<arg.min_sent_len:
                s_id=s_id+1
                continue
            if count+len_sent > arg.max_len:
                break
            else:
                sent_to_pos[str(d_id)+'-'+str(s_id)]=(count+bias,count+len_sent+bias)
                count=count+len_sent
                one_pre_src_d=one_pre_src_d+sent_encode
            s_id=s_id+1

        bias=bias+count
        one_doc_len.append(count)
        one_pre_pos_d=one_pre_pos_d+list(range(count))
            
        pre_src_d.append(one_pre_src_d)
        pre_pos_d.append(one_pre_pos_d)
        doc_len.append(one_doc_len)
        
        sent_to_pos_list.append(sent_to_pos)
    
    
    src_d, pad_mask_d=torch.tensor(pad_with_mask(pre_src_d, pad_id))  
    pos_d, pad_mask_d=torch.tensor(pad_with_mask(pre_pos_d, pad_id)) 
                
    #prepear attention mask
    mask_d=[]
    for one in doc_len:
        one_mask_d=np.zeros((len(src_d[0]), len(src_d[0])), dtype=np.float16)
        count=0
        for i in one:
            for x in range(count,count+i):
                one_mask_d[x][count:count+i]=1
            count=count+i
        mask_d.append(one_mask_d)
        
    mask_d=torch.tensor(mask_d, dtype=torch.float) 
    
    #encoder for graph
    #filter the graph

    filter_graph=[]
    for one_id,one in enumerate(graph):
        del_node=[]

        for node_id,node_content in enumerate(one.vs["position"]):
            for j in node_content:
                d=0
                s=j
                k=str(d)+'-'+str(s)            
                if k not in sent_to_pos_list[one_id].keys():
                    if k not in del_node:
                        del_node.append(node_id)

        
        new_one = copy.deepcopy(one)
        new_one.delete_vertices(del_node)
        if arg.add_root == 1:
            pass
        
        filter_graph.append(new_one)

    
    #prepear 
    src_d_merge = src_d
    max_node=max([len(one.vs['content']) for one in filter_graph])
    
    
    pre_token_to_node=[]
    pre_node_to_token=[]
    pre_node_mask=[]
    merge_tk=len(src_d_merge[0])
    for one_id,one in enumerate(filter_graph):
        #print(sent_to_pos_list[one_id])
        #print(one.vs['position'])
        one_token_to_node=[]
        one_node_to_token=[]
        one_node_mask=[]
        for node_id,node in enumerate(one.vs['content']):
            node_encode=tokenizer.encode(node,add_special_tokens=False)
            node_encode_str=[str(i) for i in node_encode]
            node_encode_str=' '.join(node_encode_str)
            mask_encode=len(node_encode)*['-1']
            mask_encode_str=' '.join(mask_encode)
            sent_pos=one.vs['position'][node_id]
            node_pos=[0]*merge_tk
            for j in sent_pos:
                d=0
                s=j
                k=str(d)+'-'+str(s)
                rangex=sent_to_pos_list[one_id][k]
                tar_sent=src_d_merge[one_id][rangex[0]:rangex[1]]
                tar_sent_str=[str(int(i)) for i in tar_sent]
                tar_sent_str=' '.join(tar_sent_str)
                tar_sent_str=tar_sent_str.replace(node_encode_str,mask_encode_str)
                tar_sent = tar_sent_str.split()
                for p,x in enumerate(tar_sent):
                    if x == '-1':
                        node_pos[p+rangex[0]]=1
            '''
            print(one.vs['content'][node_id])
            print(one.vs['position'][node_id])
            xx=[ii for ii,i in enumerate(node_pos) if i != 0]
            print(xx)
            xxx=[src_d_merge[one_id][i] for i in xx]
            print(tokenizer.decode(xxx))
            print('----------------------')
            '''
            one_node_to_token.append(node_pos)
            z=sum(node_pos)
            if z != 0:
                node_pos=[i/z for i in node_pos]
                one_token_to_node.append(node_pos)
                one_node_mask.append(1)
            else:
                one_token_to_node.append(node_pos)
                one_node_mask.append(0)                

        for i in range(max_node-len(one_node_mask)):
            one_token_to_node.append([0]*merge_tk)
            one_node_to_token.append([0]*merge_tk)
            one_node_mask.append(0)
        
        pre_token_to_node.append(one_token_to_node)
        pre_node_to_token.append(one_node_to_token)
        pre_node_mask.append(one_node_mask)  
        
    token_to_node=torch.tensor(pre_token_to_node,dtype=torch.float)
    node_to_token=torch.tensor(pre_node_to_token,dtype=torch.float)
    node_mask=torch.tensor(pre_node_mask,dtype=torch.float)    
    
    #prepear adj_graph
    adj_graph=[]
    for i,one in enumerate(filter_graph):
        one_adj=np.zeros((max_node, max_node), dtype=np.float16)
        for edge in one.get_edgelist():
            n1=edge[0]
            n2=edge[1]

            one_adj[n1][n2]=1
            one_adj[n2][n1]=1
            one_adj[n1][n1]=1  
            one_adj[n2][n2]=1  
            
        adj_graph.append(one_adj)
        
    adj_graph=torch.tensor(adj_graph, dtype=torch.float)
    
    #prepear flow_graph
    flow_graph=[]
    for i,one in enumerate(filter_graph):
        one_flow=np.zeros((max_node, max_node), dtype=np.float16)
        for edge in one.get_edgelist():
            n1=edge[0]
            n2=edge[1]
            if n1 != n2:
                #if one_flow[n1][n2] == 0 and one_flow[n2][n1] == 0:
                one_flow[n1][n2]=1
                #one_flow[n2][n1]=1
                #one_flow[n2][n1]=1*arg.one_direction_reduction 
             
        for line in range(max_node):
            sum_line=np.sum(one_flow[line])
            if sum_line > 0:
                one_flow[line]=one_flow[line]/sum_line
        flow_graph.append(one_flow)
        
    flow_graph=torch.tensor(flow_graph, dtype=torch.float)    
    
    
    #Decode
    pre_inp=[]
    pre_tar=[]
    dec_len=[]
    for one in abstract:

        one_dec_encode=[]
        
        dec_encode=tokenizer.encode(one,add_special_tokens=False)
        one_dec_encode=one_dec_encode+dec_encode

        one_pre_inp=[seq_id]+one_dec_encode
        one_pre_tar=one_dec_encode+[end_id]
        dec_len.append(min(len(one_dec_encode),arg.max_dec_steps))
        pre_inp.append(one_pre_inp)
        pre_tar.append(one_pre_tar)

        '''
        one_dec_encode=[]
        
        dec_encode=tokenizer.encode(one,add_special_tokens=False)
        one_dec_encode=one_dec_encode+dec_encode
        one_dec_encode=one_dec_encode+[end_id]

        one_pre_inp=[seq_id]+one_dec_encode
        one_pre_tar=one_dec_encode
        dec_len.append(min(len(one_dec_encode),arg.max_dec_steps))
        pre_inp.append(one_pre_inp)
        pre_tar.append(one_pre_tar)
        '''

    inp=torch.tensor(pad(pre_inp, 0)) 
    
    tar,tar_mask=torch.tensor(pad_with_mask(pre_tar, 0))  
    
    max_dec_len=max(dec_len)
    max_dec_len=torch.tensor(max_dec_len)
    dec_len=torch.tensor(dec_len)   



    pre_oracle_label=[]
    for i,one in enumerate(filter_graph):    
        one_oracle_label=one.vs['cluster']

        pre_oracle_label.append(one_oracle_label)
        
    oracle_label= torch.tensor(pad(pre_oracle_label, pad_id)) 


    tk=len(src_d[0])
    nk=len(token_to_node[0])

    src_g_add=[]
    pos_g_add=[]
    clst=[]
    clsn=[]    
    seg=[]
    for i in range(len(article_no_res)):
        src_g_add.append([cls_id]*nk)
        pos_g_add.append([200]*nk)
        seg.append([0]*tk+[1]*nk)
        clst.append(list(range(tk)))        
        clsn.append([i+tk for i in range(nk)]) 
    
    src_g_add=torch.tensor(src_g_add)
    pos_g_add=torch.tensor(pos_g_add)
    
    clst=torch.tensor(clst)
    clsn=torch.tensor(clsn)
    seg=torch.tensor(seg)
    
    src=torch.cat((src_d,src_g_add),1)
    pos=torch.cat((pos_d,pos_g_add),1)


    B,nk,tk=node_to_token.size()
    mask_add=torch.zeros((B,tk,nk),dtype=torch.float)   

    mask_1=torch.cat((mask_d,mask_add),2)
    mask_2=torch.cat((node_to_token,adj_graph),2)
    mask_final=torch.cat((mask_1,mask_2),1)

    '''
    print(adj_graph.size())
    print(mask_d.size())
    print(node_to_token.size())
    print(mask_add.size())

    print(mask_1.size())
    print(mask_2.size())
    print(mask_final.size())
    
    print(src)
    print(pos)
    print(seg)
    print(clst)
    print(clsn)
    print(mask_final)    
    torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
    print(adj_graph)
    while(True):
        pass
    '''
    pos_embedding=[]
    '''
    for one in node_to_token:
        tk=len(one[0])
        one_pos_embedding=[]
        for n in one:
            if tk >= arg.pos_dim:
                one_pos_embedding.append(n[:arg.pos_dim-1])
            else:
                one_pos_embedding.append(list(n)+[0]*(arg.pos_dim-tk))               
        pos_embedding.append(one_pos_embedding)    
    
    pos_embedding=torch.tensor(pos_embedding)
    '''

    '''
    print(abs_key[0])
    print(filter_graph[0].vs['content'])
    print(oracle_label[0])
    
    gg=copy.deepcopy(filter_graph[0])
    gg.vs['cluster'] = [int(i) for i in oracle_label[0]]
    igraph2graphviz(gg)
    '''
    #igraph2graphviz(filter_graph[0])    
    '''
    torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
    print('token_to_node:')
    print(token_to_node)
    print('node_to_token:')
    print(node_to_token)
    print(node_to_token.size())
    print('node_mask')
    print(node_mask)
    print('adj_graph')
    print(adj_graph)
    print('flow_graph')
    print(flow_graph)
    '''

    '''
    for i,one in enumerate(node_to_token[0]):
        
        xxx=filter_graph[0].vs['content'][i]
        print(xxx)   
        print(tokenizer.encode(xxx,add_special_tokens=False))
        print(one * src_d_merge[0])
    '''
    '''
    print('inp')
    print(inp)      
    print('tar')
    print(tar)      
    print('tar_mask')
    print(tar_mask)      
    print('dec_len')
    print(dec_len)      
    print('max_dec_len')
    print(max_dec_len)  
    print('doc_len')
    print(len(src_d[0])) 
    print('--------------------------')
    '''

    src_d=src_d.cuda()
    pos_d=pos_d.cuda()
    mask_d=mask_d.cuda()
    pad_mask_d=pad_mask_d.cuda()
    
    token_to_node=token_to_node.cuda()
    node_to_token=node_to_token.cuda()
    node_mask=node_mask.cuda()
    adj_graph=adj_graph.cuda()
    flow_graph=flow_graph.cuda()
    
    inp=inp.cuda()
    tar=tar.cuda()
    tar_mask=tar_mask.cuda()
    dec_len=dec_len.cuda()
    max_dec_len=max_dec_len.cuda()

    src=src.cuda()
    pos=pos.cuda()
    seg=seg.cuda()
    mask_final=mask_final.cuda()
    clst=clst.cuda()
    clsn=clsn.cuda()


    #pos_embedding=pos_embedding.cuda()
    return src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d



def data_loader_train_bert_nognn(article_no_res,abstract,graph,tokenizer,arg):
    
    cls_id=101
    seq_id=102
    mask_id=103
    end_id = 100
    pad_id=0
    dec_start_id=1
    
    #encoder for document  
    #prepear src_d  pos_d 
    pre_src_d=[]
    pre_pos_d=[]
    doc_len=[]
    
    sent_to_pos_list=[]
    


    
    
    for one in article_no_res:

        #first split for this example

        one_pre_src_d=[]
        one_pre_pos_d=[]
        one_doc_len=[]
        sent_to_pos={}
        d_id=0
        bias=0

        count=0
        s_id=0
        for sent in one:
            sent_encode=tokenizer.encode(sent,add_special_tokens=False)
            sent_encode=sent_encode+[seq_id]
            len_sent=len(sent_encode)
            if len_sent<arg.min_sent_len:
                s_id=s_id+1
                continue
            if count+len_sent > arg.max_len:
                break
            else:
                sent_to_pos[str(d_id)+'-'+str(s_id)]=(count+bias,count+len_sent+bias)
                count=count+len_sent
                one_pre_src_d=one_pre_src_d+sent_encode
            s_id=s_id+1

        bias=bias+count
        one_doc_len.append(count)
        one_pre_pos_d=one_pre_pos_d+list(range(count))
            
        pre_src_d.append(one_pre_src_d)
        pre_pos_d.append(one_pre_pos_d)
        doc_len.append(one_doc_len)
        
        sent_to_pos_list.append(sent_to_pos)
    
    
    src_d, pad_mask_d=torch.tensor(pad_with_mask(pre_src_d, pad_id))  
    pos_d, pad_mask_d=torch.tensor(pad_with_mask(pre_pos_d, pad_id)) 
                
    #prepear attention mask
    mask_d=[]
    for one in doc_len:
        one_mask_d=np.zeros((len(src_d[0]), len(src_d[0])), dtype=np.float16)
        count=0
        for i in one:
            for x in range(count,count+i):
                one_mask_d[x][count:count+i]=1
            count=count+i
        mask_d.append(one_mask_d)
        
    mask_d=torch.tensor(mask_d, dtype=torch.float) 
    
    #encoder for graph
    #filter the graph

    filter_graph=[]
    for one_id,one in enumerate(graph):
        del_node=[]

        for node_id,node_content in enumerate(one.vs["position"]):
            for j in node_content:
                d=0
                s=j
                k=str(d)+'-'+str(s)            
                if k not in sent_to_pos_list[one_id].keys():
                    if k not in del_node:
                        del_node.append(node_id)

        
        new_one = copy.deepcopy(one)
        new_one.delete_vertices(del_node)
        if arg.add_root == 1:
            pass
        
        filter_graph.append(new_one)

    
    #prepear 
    src_d_merge = src_d
    max_node=max([len(one.vs['content']) for one in filter_graph])
    
    
    pre_token_to_node=[]
    pre_node_to_token=[]
    pre_node_mask=[]
    merge_tk=len(src_d_merge[0])
    for one_id,one in enumerate(filter_graph):
        #print(sent_to_pos_list[one_id])
        #print(one.vs['position'])
        one_token_to_node=[]
        one_node_to_token=[]
        one_node_mask=[]
        for node_id,node in enumerate(one.vs['content']):
            node_encode=tokenizer.encode(node,add_special_tokens=False)
            node_encode_str=[str(i) for i in node_encode]
            node_encode_str=' '.join(node_encode_str)
            mask_encode=len(node_encode)*['-1']
            mask_encode_str=' '.join(mask_encode)
            sent_pos=one.vs['position'][node_id]
            node_pos=[0]*merge_tk
            for j in sent_pos:
                d=0
                s=j
                k=str(d)+'-'+str(s)
                rangex=sent_to_pos_list[one_id][k]
                tar_sent=src_d_merge[one_id][rangex[0]:rangex[1]]
                tar_sent_str=[str(int(i)) for i in tar_sent]
                tar_sent_str=' '.join(tar_sent_str)
                tar_sent_str=tar_sent_str.replace(node_encode_str,mask_encode_str)
                tar_sent = tar_sent_str.split()
                for p,x in enumerate(tar_sent):
                    if x == '-1':
                        node_pos[p+rangex[0]]=1
            '''
            print(one.vs['content'][node_id])
            print(one.vs['position'][node_id])
            xx=[ii for ii,i in enumerate(node_pos) if i != 0]
            print(xx)
            xxx=[src_d_merge[one_id][i] for i in xx]
            print(tokenizer.decode(xxx))
            print('----------------------')
            '''
            one_node_to_token.append(node_pos)
            z=sum(node_pos)
            if z != 0:
                node_pos=[i/z for i in node_pos]
                one_token_to_node.append(node_pos)
                one_node_mask.append(1)
            else:
                one_token_to_node.append(node_pos)
                one_node_mask.append(0)                

        for i in range(max_node-len(one_node_mask)):
            one_token_to_node.append([0]*merge_tk)
            one_node_to_token.append([0]*merge_tk)
            one_node_mask.append(0)
        
        pre_token_to_node.append(one_token_to_node)
        pre_node_to_token.append(one_node_to_token)
        pre_node_mask.append(one_node_mask)  
        
    token_to_node=torch.tensor(pre_token_to_node,dtype=torch.float)
    node_to_token=torch.tensor(pre_node_to_token,dtype=torch.float)
    node_mask=torch.tensor(pre_node_mask,dtype=torch.float)    
    
    #prepear adj_graph
    adj_graph=[]
    for i,one in enumerate(filter_graph):
        one_adj=np.zeros((max_node, max_node), dtype=np.float16)
        for edge in one.get_edgelist():
            n1=edge[0]
            n2=edge[1]

            one_adj[n1][n2]=1
            one_adj[n2][n1]=1
            one_adj[n1][n1]=1  
            one_adj[n2][n2]=1  
            
        adj_graph.append(one_adj)
        
    adj_graph=torch.tensor(adj_graph, dtype=torch.float)
    
    #prepear flow_graph
    flow_graph=[]
    for i,one in enumerate(filter_graph):
        one_flow=np.zeros((max_node, max_node), dtype=np.float16)
        for edge in one.get_edgelist():
            n1=edge[0]
            n2=edge[1]
            if n1 != n2:
                #if one_flow[n1][n2] == 0 and one_flow[n2][n1] == 0:
                one_flow[n1][n2]=1
                one_flow[n2][n1]=1
                #one_flow[n2][n1]=1*arg.one_direction_reduction 
             
        for line in range(max_node):
            sum_line=np.sum(one_flow[line])
            if sum_line > 0:
                one_flow[line]=one_flow[line]/sum_line
        flow_graph.append(one_flow)
        
    flow_graph=torch.tensor(flow_graph, dtype=torch.float)    
    
    
    #Decode
    pre_inp=[]
    pre_tar=[]
    dec_len=[]
    for one in abstract:

        one_dec_encode=[]
        
        dec_encode=tokenizer.encode(one,add_special_tokens=False)
        one_dec_encode=one_dec_encode+dec_encode

        one_pre_inp=[seq_id]+one_dec_encode
        one_pre_tar=one_dec_encode+[end_id]
        dec_len.append(min(len(one_dec_encode),arg.max_dec_steps))
        pre_inp.append(one_pre_inp)
        pre_tar.append(one_pre_tar)

        '''
        one_dec_encode=[]
        
        dec_encode=tokenizer.encode(one,add_special_tokens=False)
        one_dec_encode=one_dec_encode+dec_encode
        one_dec_encode=one_dec_encode+[end_id]

        one_pre_inp=[seq_id]+one_dec_encode
        one_pre_tar=one_dec_encode
        dec_len.append(min(len(one_dec_encode),arg.max_dec_steps))
        pre_inp.append(one_pre_inp)
        pre_tar.append(one_pre_tar)
        '''

    inp=torch.tensor(pad(pre_inp, 0)) 
    
    tar,tar_mask=torch.tensor(pad_with_mask(pre_tar, 0))  
    
    max_dec_len=max(dec_len)
    max_dec_len=torch.tensor(max_dec_len)
    dec_len=torch.tensor(dec_len)   



    pre_oracle_label=[]
    for i,one in enumerate(filter_graph):    
        one_oracle_label=one.vs['cluster']

        pre_oracle_label.append(one_oracle_label)
        
    oracle_label= torch.tensor(pad(pre_oracle_label, pad_id)) 


    tk=len(src_d[0])
    nk=len(token_to_node[0])

    src_g_add=[]
    pos_g_add=[]
    clst=[]
    clsn=[]    
    seg=[]
    for i in range(len(article_no_res)):
        src_g_add.append([cls_id]*nk)
        pos_g_add.append([200]*nk)
        seg.append([0]*tk+[1]*nk)
        clst.append(list(range(tk)))        
        clsn.append([i+tk for i in range(nk)]) 
    
    src_g_add=torch.tensor(src_g_add)
    pos_g_add=torch.tensor(pos_g_add)
    
    clst=torch.tensor(clst)
    clsn=torch.tensor(clsn)
    seg=torch.tensor(seg)
    
    src=torch.cat((src_d,src_g_add),1)
    pos=torch.cat((pos_d,pos_g_add),1)


    B,nk,tk=node_to_token.size()
    mask_add=torch.zeros((B,tk,nk),dtype=torch.float)   

    mask_1=torch.cat((mask_d,mask_add),2)
    mask_2=torch.cat((node_to_token,torch.zeros(adj_graph.size())),2)
    mask_final=torch.cat((mask_1,mask_2),1)

    '''
    print(adj_graph.size())
    print(mask_d.size())
    print(node_to_token.size())
    print(mask_add.size())

    print(mask_1.size())
    print(mask_2.size())
    print(mask_final.size())
    
    print(src)
    print(pos)
    print(seg)
    print(clst)
    print(clsn)
    print(mask_final)    
    torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
    print(adj_graph)
    while(True):
        pass
    '''
    pos_embedding=[]
    '''
    for one in node_to_token:
        tk=len(one[0])
        one_pos_embedding=[]
        for n in one:
            if tk >= arg.pos_dim:
                one_pos_embedding.append(n[:arg.pos_dim-1])
            else:
                one_pos_embedding.append(list(n)+[0]*(arg.pos_dim-tk))               
        pos_embedding.append(one_pos_embedding)    
    
    pos_embedding=torch.tensor(pos_embedding)
    '''

    '''
    print(abs_key[0])
    print(filter_graph[0].vs['content'])
    print(oracle_label[0])
    
    gg=copy.deepcopy(filter_graph[0])
    gg.vs['cluster'] = [int(i) for i in oracle_label[0]]
    igraph2graphviz(gg)
    '''
    #igraph2graphviz(filter_graph[0])    
    '''
    torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
    print('token_to_node:')
    print(token_to_node)
    print('node_to_token:')
    print(node_to_token)
    print(node_to_token.size())
    print('node_mask')
    print(node_mask)
    print('adj_graph')
    print(adj_graph)
    print('flow_graph')
    print(flow_graph)
    '''

    '''
    for i,one in enumerate(node_to_token[0]):
        
        xxx=filter_graph[0].vs['content'][i]
        print(xxx)   
        print(tokenizer.encode(xxx,add_special_tokens=False))
        print(one * src_d_merge[0])
    '''
    '''
    print('inp')
    print(inp)      
    print('tar')
    print(tar)      
    print('tar_mask')
    print(tar_mask)      
    print('dec_len')
    print(dec_len)      
    print('max_dec_len')
    print(max_dec_len)  
    print('doc_len')
    print(len(src_d[0])) 
    print('--------------------------')
    '''

    src_d=src_d.cuda()
    pos_d=pos_d.cuda()
    mask_d=mask_d.cuda()
    pad_mask_d=pad_mask_d.cuda()
    
    token_to_node=token_to_node.cuda()
    node_to_token=node_to_token.cuda()
    node_mask=node_mask.cuda()
    adj_graph=adj_graph.cuda()
    flow_graph=flow_graph.cuda()
    
    inp=inp.cuda()
    tar=tar.cuda()
    tar_mask=tar_mask.cuda()
    dec_len=dec_len.cuda()
    max_dec_len=max_dec_len.cuda()

    src=src.cuda()
    pos=pos.cuda()
    seg=seg.cuda()
    mask_final=mask_final.cuda()
    clst=clst.cuda()
    clsn=clsn.cuda()


    #pos_embedding=pos_embedding.cuda()
    return src,pos,seg,mask_final,clst,clsn,pad_mask_d,token_to_node,node_to_token,node_mask,adj_graph,flow_graph,inp,tar,tar_mask,dec_len,max_dec_len,oracle_label,pos_embedding,src_d




from graphviz import Digraph    
def igraph2graphviz(g):
    dot = Digraph(engine='neato')
    
    colorl=['black','red']
    for i in range(len(g.vs)):
        content=g.vs['content'][i]
        colorx=g.vs['cluster'][i]       
        dot.node(str(i), content, color=colorl[colorx])
    for i in g.get_edgelist():
        dot.edge(str(i[0]), str(i[1]), constraint='false')
    
#    dot.render('round-table.gv', view=False)
    print(dot)
    return dot   




def data_loader_train():
    pass

def data_loader_train_split():
    pass







def data_loader_train_split_doc(article_no_res,abstract,graph,tokenizer,arg):
    cls_id=101
    seq_id=102
    mask_id=103
    pad_id=0
    dec_start_id=1
    end_id = 100
    
    #encoder for document  
    #we split each example to two encoder, so the bacth size for bert is bacth size *2
    #prepear src_d  pos_d 
    pre_src_d=[]
    pre_pos_d=[]
    doc_len=[]
    
    sent_to_pos_list=[]
    
    for one in article_no_res:

        #first split for this example

        one_pre_src_d=[]
        one_pre_pos_d=[]
        one_doc_len=[]
        sent_to_pos={}
        d_id=0
        bias=0

        count=0
        s_id=0
        for sent in one:
            sent_encode=tokenizer.encode(sent,add_special_tokens=False)
            sent_encode=sent_encode+[seq_id]
            len_sent=len(sent_encode)
            if len_sent<arg.min_sent_len:
                s_id=s_id+1
                continue
            if count+len_sent > arg.max_len:
                break
            else:
                sent_to_pos[str(d_id)+'-'+str(s_id)]=(count+bias,count+len_sent+bias)
                count=count+len_sent
                one_pre_src_d=one_pre_src_d+sent_encode
            s_id=s_id+1

        bias=bias+count
        one_doc_len.append(count)
        one_pre_pos_d=one_pre_pos_d+list(range(count))
            
        pre_src_d.append(one_pre_src_d)
        pre_pos_d.append(one_pre_pos_d)
        doc_len.append(one_doc_len)
        
        sent_to_pos_list.append(sent_to_pos)
        

        
    src_d, pad_mask_d=torch.tensor(pad_with_mask(pre_src_d, 0))  
    pos_d, pad_mask_d=torch.tensor(pad_with_mask(pre_pos_d, 0)) 


    #prepear attention mask
    
    mask_d=[]
    for one in doc_len:
        one_mask_d=np.zeros((len(src_d[0]), len(src_d[0])), dtype=np.float16)
        count=0
        for i in one:
            for x in range(count,count+i):
                one_mask_d[x][count:count+i]=1
            count=count+i
        mask_d.append(one_mask_d)
        
    mask_d=torch.tensor(mask_d, dtype=torch.float) 
    
    #Decode
    pre_inp=[]
    pre_tar=[]
    dec_len=[]
    for one in abstract:
        one_dec_encode=[]
        
        dec_encode=tokenizer.encode(one,add_special_tokens=False)
        one_dec_encode=one_dec_encode+dec_encode
        one_dec_encode=one_dec_encode+[end_id]

        one_pre_inp=[seq_id]+one_dec_encode
        one_pre_tar=one_dec_encode
        dec_len.append(min(len(one_dec_encode),arg.max_dec_steps))
        pre_inp.append(one_pre_inp)
        pre_tar.append(one_pre_tar)
    

    inp=torch.tensor(pad(pre_inp, 0)) 
    
    tar,tar_mask=torch.tensor(pad_with_mask(pre_tar, 0))  
    
    max_dec_len=max(dec_len)
    max_dec_len=torch.tensor(max_dec_len)
    dec_len=torch.tensor(dec_len)    
    
    
    
    src_d=src_d.cuda()
    pos_d=pos_d.cuda()
    mask_d=mask_d.cuda()
    pad_mask_d=pad_mask_d.cuda()
    
    
    inp=inp.cuda()
    tar=tar.cuda()
    tar_mask=tar_mask.cuda()
    dec_len=dec_len.cuda()
    max_dec_len=max_dec_len.cuda()
    '''
    print('src_d')
    print(src_d) 
    print('pos_d')
    print(pos_d) 
    print('mask_d')
    print(mask_d)     
    print('pad_mask_d')
    print(pad_mask_d)     
    
    print('inp')
    print(inp)      
    print('tar')
    print(tar)      
    print('tar_mask')
    print(tar_mask)      
    print('dec_len')
    print(dec_len)      
    print('max_dec_len')
    print(max_dec_len)  
    print('doc_len')
    print(len(src_d[0])) 
    
    while(True):
        pass
    '''
    return src_d,pos_d,mask_d,pad_mask_d,inp,tar,tar_mask,dec_len,max_dec_len

                    





def fact_transform(source,tar):

    cls_id=torch.tensor(101)
    seq_id=torch.tensor(102)
    mask_id=103
    pad_id=0
    dec_start_id=1
    end_id = 100
    
    source=source.cpu()
    tar=tar.cpu()
    
    #prepear src_d  pos_d 
    pre_src_d=[]
    pre_pos_d=[]
    pre_seg_d=[]
    doc_len=[]
    


    one_pre_src_d=[]
    one_pre_pos_d=[]
    one_pre_seg_d=[]
    
    source_encode=list(source)
    claim_encode=list(tar)
    
    source_encode=source_encode[:200]
    claim_encode=claim_encode[:50]
    
    while(torch.tensor(102) in source_encode):
        source_encode.remove(torch.tensor(102))
    
    one_pre_src_d=one_pre_src_d+[cls_id]
    
    one_pre_src_d=one_pre_src_d+source_encode+[seq_id]
    
    one_pre_src_d=one_pre_src_d+claim_encode+[seq_id]  
    
    one_pre_pos_d=one_pre_pos_d+list(range(len(one_pre_src_d))) 
    
    one_pre_seg_d=one_pre_seg_d+[0]*(1+len(source_encode)+1)+[1]*(len(claim_encode)+1)


    pre_src_d.append(one_pre_src_d)
    pre_pos_d.append(one_pre_pos_d)
    pre_seg_d.append(one_pre_seg_d)
    doc_len.append(len(one_pre_seg_d))

        
    src, pad_mask=torch.tensor(pad_with_mask(pre_src_d, 0))  
    pos, pad_mask=torch.tensor(pad_with_mask(pre_pos_d, 0)) 
    seg, pad_mask=torch.tensor(pad_with_mask(pre_seg_d, 0)) 

    
    mask_d=[]
    for one in range(len(src)):
        one_mask_d=np.zeros((len(src[0]), len(src[0])), dtype=np.float16)
        for i in range(doc_len[one]):
            one_mask_d[i][0:doc_len[one]]=1
            
        mask_d.append(one_mask_d)
        
    mask_d=torch.tensor(mask_d, dtype=torch.float) 
    
    

    
    src=src.cuda()
    pos=pos.cuda()
    seg=seg.cuda()
    mask_d=mask_d.cuda()
    pad_mask=pad_mask.cuda()
    
    clss=[]
    for one in src:
        clss.append([0])
    
    clss=torch.tensor(clss)
    clss=clss.cuda()  


    return src,pos,seg,mask_d,pad_mask,clss










