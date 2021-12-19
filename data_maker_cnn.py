# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:04:28 2020

@author: User
"""
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import igraph as ig
from graphviz import Digraph
import numpy
import copy
import time
import torch
import random
import pickle
    
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

def sent_sim(s1,s2):
    from rouge import Rouge    
    raw_rouge=Rouge()    
    scoresx = raw_rouge.get_scores(s1, s2)
    r1=scoresx[0]['rouge-1']['p']
    r2=scoresx[0]['rouge-2']['f']
    return r1    

def key_sim(s1,s2):
    k1=getkey(s1)
    k2=getkey(s2)
    recall=0
    precision=0
    for i in k1:
        if i in k2:
            precision=precision+1
    for i in k2:
        if i in k1:
            recall=recall+1    
    recall=recall/len(k2)
    precision=precision/len(k1)
    return recall  



def wordlen(s):
    return len(s.split())

def getkey(s):
    k=[]
    for i in s.split():
        if i not in stop_word and i not in k:
            k.append(i)
    
    return k
            
            
def take2(elem):
    return elem[1]            

def take4(elem):
    return elem[3] 






#----------------------Co-reference Resolution--------------------------------
def coreference_resolution(doc,nlp):

    result=nlp.predict(document=doc)  

    token=result['document']
    cluster=result['clusters']
    
    res_cluster=[]
    for i in cluster:
        one_res_cluster=[]
        for j in i:
            one=''
            for w in token[j[0]:j[1]+1]:
                one = one + w +' '
            one = one.strip()
            one_res_cluster.append(one)
        res_cluster.append(one_res_cluster)
        
        
    lenth=len(token)
    pos_to_word={}
    for i,itoken in enumerate(token):
        stri=str(i)
        while(len(stri)<=4):
            stri='0'+stri
        pos_to_word[stri]=itoken
        
    token_pos=list_to_string(list(range(lenth)))
    
    cluster_pos=[]
    for one_cluster in cluster:
        one_cluster_pos=[]
        for entity in one_cluster:
            start=entity[0]
            end=entity[1]
            one_cluster_pos.append(list_to_string(list(range(start,end+1))))
        cluster_pos.append(one_cluster_pos)
    
    entity_list=[]
    for one_cluster in cluster_pos:
        for entity in one_cluster:
            entity_list.append(entity)
            
            
    allow_entity_list=[]
    for i,ientity in enumerate(entity_list):
        entity=[]
        for w in ientity.split():
            entity.append(pos_to_word[w])
        e_len,_=count_without_stop(entity)
        if e_len == 0:
            allow_entity_list.append(ientity)
        
        
    for one_cluster in cluster_pos:
        target=''
        len_min=999
        for e in one_cluster:
            if e not in allow_entity_list:
                e_len=len(e.split())
                if e_len < len_min:
                    len_min=e_len
                    target=e
        '''
        p=''
        for i in target.split():
            p=p+pos_to_word[i]+' ' 
        print(p)
        '''
        if target != '':
            for other in one_cluster:
                if other in allow_entity_list:
                    token_pos=token_pos.replace(other,target)
        
    doc_new=''
    for i in token_pos.split():
        doc_new=doc_new+pos_to_word[i]+' '
    
    return doc_new.strip()

        
def list_to_string(inlist):
    s=''
    for i in inlist:
        stri=str(i)
        while(len(stri)<=4):
            stri='0'+stri
        s=s+stri+' '
    return s.strip()

#------------------------allen------------------------------

def make_relation_allen(sent,nlp,bias,idx_bias):

    x=nlp.predict(sentence=sent)

    verbs=x['verbs']
    lens=len(x['words'])
    
    relation=[]
    for i in verbs:
        
        '''
        pos_dict={}
        for p,t in enumerate(i['tags']):
            if t != '0':
                tt=t[2:]
                if tt not in pos_dict.keys():
                    pos_dict[tt]=[p+bias]
                else:
                    pos_dict[tt]=pos_dict[tt]+[p+bias]
        '''
        
        seq=i['description']
        tag_list=i['tags']
        tag1=[]
        tag2=[]
        for num,char in enumerate(seq):
            if char == '[':
                tag1.append(num)
            if char == ']':
                tag2.append(num)    
        assert len(tag1)==len(tag2)
        
        fact=[]
        for num,pos in enumerate(tag1):
            fact.append(seq[pos:tag2[num]+1])

        true_fact=[]
        for entity in fact:
            entity=entity.strip('[')
            entity=entity.strip(']')
            try:
                tag,string=entity.split(': ')
            except:
                print(entity)
            #filter with content
            #if string in []:
                #true_fact=[]
                #break
            #filter with tags
            if tag not in ['ARGM-ADV','ARGM-MNR','C-ARG1','C-ARG0','C-ARG2','C-ARG3','ARGM-DIS','ARGM-MOD']:
            #filter with length
                if len(string.split()) <= 20:
                    idx_list=[]
                    for one_tag_id,one_tag in enumerate(tag_list):
                        if tag==one_tag[2:]:
                            idx_list.append(one_tag_id+idx_bias)
                    
                    true_fact.append((tag+':'+string, [bias], idx_list))

        #filter with length
        if len(true_fact) <= 2:
            continue
        
       
        relation.append(true_fact)
    
        
    return fact_merge(relation),lens


def fact_merge(rlist):
    merged={}
    for i in rlist:
        fact=''
        for j in i:
            tag,string=j[0].split(':')            
            fact=fact+string+' '
        merged[fact]=i
        
    select=[]
    key=list(merged.keys())
    for i in range(len(key)):
        tag=0
        for j in range(len(key)):
            if i == j:
                continue
            if include(key[i],key[j]):
                tag=1
                break
        if tag == 0:
            select.append(key[i])
    return_list=[]
    for i in select:
        return_list.append(merged[i])
    return return_list

def include(s1,s2):

    l1=s1.split()
    l2=s2.split()
    len1,clear1=count_without_stop(l1)
    len2,clear2=count_without_stop(l2)
    common=LCS(clear1,clear2)
    difference=len1-common
    if len1==0:
        return False
    else:
        if difference == 0 and len1 <=4:
            return True
        if difference <=1 and len1 >4:
            return True
    return False
#------------------------make graph with allennlp------------------------------

def make_graph_allen(tri,coreference_chain):
#    print(tri)
    g = ig.Graph(directed=True)
    
    
    dot = Digraph(engine='neato')
#    dot.attr(rankdir='LR', size='8,8', overlap='false')
    dot.attr(overlap='false')
    edges=[]
    nodes=[]
    #build nodes
    #node with tags in no_merge list will not be merged together even they are the same    
    
    no_merge_tag=['V','ARGM-NEG']
    no_merge_string=stop_word
    fact_to_node={}
    node_to_fact={}
    node_to_pos={}
    node_to_idx={}
    count=0
    objects=[]
    for s in tri:
        for f in s:
            fact=''
            for obj in f:
                tag,string=obj[0].split(':')
                fact=fact+string+' '
                
            for obj in f:
                tag,string=obj[0].split(':')
                pos=obj[1]
                idx=obj[2]
                if tag not in no_merge_tag and string not in no_merge_string:
                    if string not in objects:
                        objects.append(string)
            
                    if string not in fact_to_node.keys():
                        fact_to_node[string]='N'+str(count)
                        node_to_fact['N'+str(count)]=string
                        node_to_pos['N'+str(count)]=pos
                        node_to_idx['N'+str(count)]=idx
                        
                        dot.node('N'+str(count), string)
                        nodes.append('N'+str(count))
                        count=count+1 
                    else:
                        n=fact_to_node[string]
                        node_to_pos[n]=node_to_pos[n]+pos
                        node_to_idx[n]=node_to_idx[n]+idx

                else:
                    fact_to_node[fact+string]='N'+str(count)    
                    node_to_fact['N'+str(count)]=string
                    node_to_pos['N'+str(count)]=pos
                    node_to_idx['N'+str(count)]=idx
                    dot.node('N'+str(count), string)
                    nodes.append('N'+str(count))
                    count=count+1                    
    
    g.add_vertices(len(nodes))
    content=[]
    position=[]
    idx_list=[]
    for i in nodes:
        content.append(node_to_fact[i])
        position.append(node_to_pos[i])
        idx_list.append(node_to_idx[i])
    g.vs["content"] = content
    g.vs["position"] = position
    g.vs["idx"]=idx_list

    edges_label=[]
    #build graph based on open IE relation
    for s in tri:
        for f in s:             
            fact=''
            for obj in f:
                tag,string=obj[0].split(':')
                fact=fact+string+' '
            '''
            #one approach for adding edges
            for i in range(len(f)):
                for j in range(len(f)):
                    if i == j:
                        continue
                    tag1,string1=f[i][0].split(':')
                    tag2,string2=f[j][0].split(':')    

                    if tag1 == 'ARG0' and tag2 == 'V':
                    
                        if tag1 not in no_merge_tag and string1 not in no_merge_string:
                            n1=fact_to_node[string1]
                        else:
                            n1=fact_to_node[fact+string1]
                            
                        if tag2 not in no_merge_tag and string2 not in no_merge_string:
                            n2=fact_to_node[string2]
                        else:
                            n2=fact_to_node[fact+string2]                
        
                        dot.edge(n1, n2, constraint='false') 
                        edges.append((n1,n2))
                        edges_label.append(0)
                        
                        
                    if tag1 == 'V' and tag2 != 'ARG0':
                    
                        if tag1 not in no_merge_tag and string1 not in no_merge_string:
                            n1=fact_to_node[string1]
                        else:
                            n1=fact_to_node[fact+string1]
                            
                        if tag2 not in no_merge_tag and string2 not in no_merge_string:
                            n2=fact_to_node[string2]
                        else:
                            n2=fact_to_node[fact+string2]                
        
                        dot.edge(n1, n2, constraint='false') 
                        edges.append((n1,n2))
                        edges_label.append(0)

            '''
            #one approach for adding edges                

            for i in range(len(f)):
                if i == len(f)-1:
                    break
                tag1,string1=f[i][0].split(':')
                tag2,string2=f[i+1][0].split(':')                
                
                if tag1 not in no_merge_tag and string1 not in no_merge_string:
                    n1=fact_to_node[string1]
                else:
                    n1=fact_to_node[fact+string1]
                    
                if tag2 not in no_merge_tag and string2 not in no_merge_string:
                    n2=fact_to_node[string2]
                else:
                    n2=fact_to_node[fact+string2]                

                dot.edge(n1, n2, constraint='false') 
                edges.append((n1,n2))
                edges_label.append(0)

    #add additional edges based on simialarity

    for fact1 in objects:
        for fact2 in objects: 
            if fact1==fact2:
                continue
            else:
                if add_egde(fact1,fact2):
                    n1=fact_to_node[fact1]
                    n2=fact_to_node[fact2]    
                    dot.edge(n1, n2, constraint='false')    
                    edges.append((n1,n2))
                    edges_label.append(1)
                    
    #add additional edges based on coreference
    #for cluster in coreference_chain:
    for i in coreference_chain.keys():
        one_cluster=coreference_chain[i]
        one_cluster_pos=[]
        for j in one_cluster:
            one_cluster_pos.append(list(range(j['start'],j['end'])))
            
        
        one_cluster_node=[]
        for node_id, one_idx in enumerate(g.vs["idx"]):
            for one_entity in one_cluster_pos:
                if list_in(one_entity, one_idx):
                    one_cluster_node.append('N'+str(node_id))
                    break

        for node1 in one_cluster_node:
            for node2 in one_cluster_node: 
                if node1==node2:
                    continue
                else:                 
                    dot.edge(node1, node2, constraint='false')    
                    edges.append((node1,node2))
                    edges_label.append(2)
                    dot.edge(node2, node1, constraint='false')    
                    edges.append((node2,node1))
                    edges_label.append(2)
                    
    edges=list(set(edges))
    g_edges=[]          
    for i in edges:
        x1=int((i[0].split('N'))[1])
        x2=int((i[1].split('N'))[1])
        g_edges.append((x1,x2))
    g_edges=list(set(g_edges))
    g.add_edges(g_edges)
    g.es["label"] = edges_label
   
    dele=[]
    for i,num in enumerate(g.degree()):
        if num == 0:
            dele.append(i)

    g.delete_vertices(dele)

    gg=g.community_label_propagation()

    
    
    '''
    for i in range(len(g.vs["content"])):
        print(g.vs["content"][i])
        print(g.vs["position"][i]) 
        print('----------')
    '''    
        
    colorl=['red','green','yellow','blue','lightgrey','lightblue2','grey']
    for i in range(gg.__len__()):
        if gg[i] == [0]:
            continue
        else:
            for j in gg[i]:
                dot.node(name=('N'+str(j)),color=colorl[i%len(colorl)])
    
#    dot.render('round-table.gv', view=False)

    return g

#longest commone sequence 
def LCS(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
            else:
                res[i][j] = max(res[i-1][j],res[i][j-1])
    return res[-1][-1]


#function that clear the stopword in the sequence
def count_without_stop(rlist):
    stop_list=stop_word 
    count=0
    clear=[]
    for i in rlist:
        if i not in stop_list:
            count=count+1
            clear.append(i)
    return count,clear

#the function that determine whether two nodes need to be connected
#You can modified the function 
#the input is two string, s1, s2, from two nodes
#the output is bool to determine whether two nodes need to be connected
def add_egde(s1,s2):
    l1=getkey(s1)
    l2=getkey(s2)

    common=0
    for k in l1:
        if k in l2:
            common=common+1
    min_len=min(len(l1), len(l2))
    if min_len>3:
        if common >=2:
            return True
        else:
            return False 
    else:
        if common >=1:
            return True
        else:
            return False         

def list_in(l1,l2):
    tag=0
    for i in l1:
        if i not in l2:
            tag=1
            break
        
    if tag==0:
        return True
    else:
        return False


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
#-----------------------------------------------------------------------------------

def coreference_convert(doc, chain):
    doc_token=doc.split()
    for i in chain.keys():
        one_cluster=chain[i]
        one_cluster_word=[]
        for j in one_cluster:
            one_cluster_word.append(doc_token[j['start']:j['end']])
        
def chain_to_relation(chain, article):
    article_token=article.split(' ')
    mid_pos=article_token.index('.')
    
    output=[]
    
    for i in chain.keys():
        one_cluster=chain[i]
        
        one_relation=[]
        for j in one_cluster:
            
            pos=list(range(j['start'],j['end']))
            
            if j['end'] <= mid_pos:
                sent=[0]
            else:
                sent=[1]
            
            content=' '.join(article_token[j['start']:j['end']])
            content='coreference:'+content
            one_relation.append((content,sent,pos))
            
        one_relation.sort(key=takekey)
        one_relation.reverse()
        
        output.append(one_relation)
        
    return [output]

def takekey(elem):
    return elem[2][0]   
#-----------------------------------------------------------------------------------
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
#x=make_relation_allen("In December, John decided to join the party .", predictor,0,0)


data_path='data/input_decoding_heuristicset/test.tsv'

import csv

all_data=[]
with open(data_path, encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        all_data.append(line)

all_data=all_data[1:]

countrecall = []

countx = 0
county = 0

id_article=0
id_fusion=0
batch_count=0
b_c=0
article_b=[]
abstract_b=[]
graph_b=[]
other_b=[]
document_id_b=[]
raw_article_b=[]

for t in range(len(all_data)): 

    if batch_count == 32:
        
        batch={}
        batch['article']=article_b
        batch['abstract']=abstract_b            
        batch['graph']=graph_b            
        batch['other']=other_b            
        batch['document']=document_id_b
        batch['raw_article']=raw_article_b

        f=open('data_file/cnn_best2/test/'+str(b_c)+'_test_batch_of '+str(len(article_b))+' examples.pkl','wb')  
        pickle.dump(batch,f)  
        f.close() 

        print("finish "+str(b_c)+" batch")
        
        
        batch_count=0
        article_b=[]
        abstract_b=[]
        graph_b=[]
        other_b=[]
        document_id_b=[]
        raw_article_b=[]
        
        b_c=b_c+1

    try:
    
        countx=countx+1
        
        article = all_data[t][0].replace(':', ',').replace('``', "''").lower()
        abstract = all_data[t][1].replace('``', "''")
        chain = eval(all_data[t][5])
        tar_k=getkey(abstract)
    
    
        new_article=[]
        for k in article.split(' . '):
            if len(k) > 3:
                new_article.append(k)
        
    
        #new_article=doc_new
        relation=[]
        bias=0
        idx_bias=0
        for j in new_article: 
            j=j.strip()
            one,add_bias=make_relation_allen(j,predictor,bias,idx_bias) 
            relation.append(one)
            bias=bias+1
            idx_bias=idx_bias+add_bias+1

        graph=make_graph_allen(relation, chain) 

        if len(graph.get_edgelist()) == 0:
            relation=chain_to_relation(chain,article)
            graph=make_graph_allen(relation, {})   
        
        one_oracle_label=[]
        for node in graph.vs['content']:
            tag=0
            for w in node.split():
                if w in tar_k:
                    tag=tag+1
            if tag >= 1:
                one_oracle_label.append(1)
            else:
                one_oracle_label.append(0)                    
        
        graph.vs['cluster'] = one_oracle_label
        '''
        print(article)
        print(chain)
        igraph2graphviz(graph) 
        print('---------------')
        '''
        gra_k=[]
        for n in graph.vs["content"]:
            gra_k=gra_k+getkey(n)
        
        cc=0
        for k in tar_k:
            if k in gra_k:
                cc=cc+1
        
        countrecall.append(cc/len(tar_k))
            
            
        if len(graph.get_edgelist()) == 0:         
            print('error')
            #print('article: ',article)
            #print('relation: ',relation)
            #print('abstract: ',abstract)
            #print('chain: ',chain)
            #print('--------------------------------')
            continue
            
        article_b.append(new_article)
        raw_article_b.append(new_article)
        abstract_b.append(abstract)
        graph_b.append(graph)
        other_b.append(new_article)
        document_id_b.append(id_article)           
        
    
        id_fusion=id_fusion+1
        batch_count=batch_count+1
            
        id_article=id_article+1
        
    except:
        print('one_fail')
        #print('article: ',article)
        #print('abstract: ',abstract)
        #print('chain: ',chain)
        #print('--------------------------------')
#    for x,node in enumerate(graph.vs["content"]):
#        print(node,graph.vs["idx"][x])
       
            
print(countx)
print(county)
print(sum(countrecall)/len(countrecall))

                
