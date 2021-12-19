import torch
import numpy as np
import pickle
import copy
import time
import glob
import fact_trans 


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



def data_loader_train(article_no_res,abstract,tokenizer,arg,trans):
    cls_id=101
    seq_id=102
    mask_id=103
    pad_id=0
    dec_start_id=1
    end_id = 100
    
    input_seq=[]
    labels=[]
    methods=[]
    for i in range(len(article_no_res)):

        raw_source=' '
        for j in article_no_res[i]:
            raw_source=raw_source+j+' '
        raw_source=raw_source.strip()
        raw_claim=abstract[i]
        source,claim,one_label,method=trans.sample_tran(raw_source,raw_claim)
        input_seq.append((source,claim))
        labels.append(one_label)
        methods.append(method)

    #prepear src_d  pos_d 
    pre_src_d=[]
    pre_pos_d=[]
    pre_seg_d=[]
    doc_len=[]
    
    for one in input_seq:

        #first split for this example

        one_pre_src_d=[]
        one_pre_pos_d=[]
        one_pre_seg_d=[]
        
        source_encode=tokenizer.encode(one[0],add_special_tokens=False)
        claim_encode=tokenizer.encode(one[1],add_special_tokens=False)
        
        source_encode=source_encode[:200]
        claim_encode=claim_encode[:60]
        
        
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

    
    labels=torch.tensor(labels)
    #prepear attention mask
    
    mask_d=[]
    for one in range(len(input_seq)):
        one_mask_d=np.zeros((len(src[0]), len(src[0])), dtype=np.float16)
        for i in range(doc_len[one]):
            one_mask_d[i][0:doc_len[one]]=1
            
        mask_d.append(one_mask_d)
        
    mask_d=torch.tensor(mask_d, dtype=torch.float) 
    
    
    clss=[]
    for one in input_seq:
        clss.append([0])
    
    clss=torch.tensor(clss)
  
    src=src.cuda()
    pos=pos.cuda()
    seg=seg.cuda()
    mask_d=mask_d.cuda()
    pad_mask=pad_mask.cuda()
    
    labels=labels.cuda()
    
    clss=clss.cuda()


    return src,pos,seg,mask_d,pad_mask,labels,clss,methods

                    


















