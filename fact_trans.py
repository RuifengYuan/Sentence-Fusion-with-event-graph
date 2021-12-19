# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:53:32 2020

@author: User
"""
import glob
import pickle
import random
import spacy

def data_loader(data_path):
    
    seq=[]
    
    tar=[]
    
    test_filelist = glob.glob(data_path)
    for batch_path in test_filelist:
        f=open(batch_path,'rb')
        one_batch= pickle.load(f)
        
        article=one_batch['article']
        abstract=one_batch['abstract']
        
        seq=seq+article
        
        tar=tar+abstract
            
    return seq,tar




def align_ws(old_token, new_token):
    # Align trailing whitespaces between tokens
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token


def make_new_example(eid=None, text=None, claim=None, label=None, extraction_span=None,
                     backtranslation=None, augmentation=None, augmentation_span=None, noise=None):
    # Embed example information in a json object.
    return {
        "id": eid,
        "text": text,
        "claim": claim,
        "label": label,
        "extraction_span": extraction_span,
        "backtranslation": backtranslation,
        "augmentation": augmentation,
        "augmentation_span": augmentation_span,
        "noise": noise
    }




LABEL_MAP = {True: "CORRECT", False: "INCORRECT"}

class Transformation():
    # Base class for all data transformations

    def __init__(self):
        # Spacy toolkit used for all NLP-related substeps
        self.spacy = spacy.load("en_core_web_sm")

    def transform(self, example):
        # Function applies transformation on passed example
        pass
    
class PronounSwap(Transformation):
    # Swap randomly chosen pronoun
    def __init__(self, prob_swap=0.5):
        super().__init__()

        self.class2pronoun_map = {
            "SUBJECT": ["you", "he", "she", "we", "they"],
            "OBJECT": ["me", "you", "him", "her", "us", "them"],
            "POSSESSIVE": ["my", "your", "his", "her", "its", "out", "your", "their"],
            "REFLEXIVE": ["myself", "yourself", "himself", "itself", "outselves", "yourselves", "themselves"]
        }

        self.pronoun2class_map = {pronoun: key for (key, values) in self.class2pronoun_map.items() for pronoun in values}
        self.pronouns = {pronoun for (key, values) in self.class2pronoun_map.items() for pronoun in values}

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim, aug_span = self.__swap_pronouns(new_example["claim"])

        if new_claim:
            new_example["claim"] = new_claim
            new_example["label"] = LABEL_MAP[False]
            new_example["augmentation"] = self.__class__.__name__
            new_example["augmentation_span"] = aug_span
            return new_example
        else:
            return None

    def __swap_pronouns(self, claim):
        # find pronouns
        claim_pronouns = [token for token in claim if token.text.lower() in self.pronouns]

        if not claim_pronouns:
            return None, None

        # find pronoun replacement
        chosen_token = random.choice(claim_pronouns)
        chosen_ix = chosen_token.i
        chosen_class = self.pronoun2class_map[chosen_token.text.lower()]

        candidate_tokens = [token for token in self.class2pronoun_map[chosen_class] if token != chosen_token.text.lower()]

        if not candidate_tokens:
            return None, None

        # swap pronoun and update indices
        swapped_token = random.choice(candidate_tokens)
        swapped_token = align_ws(chosen_token.text_with_ws, swapped_token)
        swapped_token = swapped_token if chosen_token.text.islower() else swapped_token.capitalize()

        claim_tokens = [token.text_with_ws for token in claim]
        claim_tokens[chosen_ix] = swapped_token

        # create new claim object
        new_claim = self.spacy("".join(claim_tokens))
        augmentation_span = (chosen_ix, chosen_ix)

        if claim.text == new_claim.text:
            return None, None
        else:
            return new_claim, augmentation_span


class NERSwap(Transformation):
    # Swap NER objects - parent class
    def __init__(self):
        super().__init__()
        self.categories = ()

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim, aug_span = self.__swap_entities(new_example["text"], new_example["claim"])

        if new_claim:
            new_example["claim"] = new_claim
            new_example["label"] = LABEL_MAP[False]
            new_example["augmentation"] = self.__class__.__name__
            new_example["augmentation_span"] = aug_span
            return new_example
        else:
            return None

    def __swap_entities(self, text, claim):
        # find entities in given category
        text_ents = [ent for ent in text.ents if ent.label_ in self.categories]
        claim_ents = [ent for ent in claim.ents if ent.label_ in self.categories]

        if not claim_ents or not text_ents:
            return None, None

        # choose entity to replace and find possible replacement in source
        replaced_ent = random.choice(claim_ents)
        candidate_ents = [ent for ent in text_ents if ent.text != replaced_ent.text and ent.text not in replaced_ent.text and replaced_ent.text not in ent.text]

        if not candidate_ents:
            return None, None

        # update claim and indices
        swapped_ent = random.choice(candidate_ents)
        claim_tokens = [token.text_with_ws for token in claim]
        swapped_token = align_ws(replaced_ent.text_with_ws, swapped_ent.text_with_ws)
        claim_swapped = claim_tokens[:replaced_ent.start] + [swapped_token] + claim_tokens[replaced_ent.end:]

        # create new claim object
        new_claim = self.spacy("".join(claim_swapped))
        augmentation_span = (replaced_ent.start, replaced_ent.start + len(swapped_ent) - 1)

        if new_claim.text == claim.text:
            return None, None
        else:
            return new_claim, augmentation_span


class EntitySwap(NERSwap):
    # NER swapping class specialized for entities (people, companies, locations, etc.)
    def __init__(self):
        super().__init__()
        self.categories = ("PERSON", "ORG", "NORP", "FAC", "GPE", "LOC", "PRODUCT",
                           "WORK_OF_ART", "EVENT")


class NumberSwap(NERSwap):
    # NER swapping class specialized for numbers (excluding dates)
    def __init__(self):
        super().__init__()

        self.categories = ("PERCENT", "MONEY", "QUANTITY", "CARDINAL")


class DateSwap(NERSwap):
    # NER swapping class specialized for dates and time
    def __init__(self):
        super().__init__()

        self.categories = ("DATE", "TIME")


class AddNoise(Transformation):
    # Inject noise into claims
    def __init__(self, noise_prob=1, delete_prob=0.5):
        super().__init__()

        self.noise_prob = noise_prob
        self.delete_prob = delete_prob
        self.spacy = spacy.load("en_core_web_sm")

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        claim = new_example["claim"]
        aug_span = new_example["augmentation_span"]
        new_claim, aug_span = self.__add_noise(claim, aug_span)

        if new_claim:
            new_example["claim"] = new_claim
            new_example["augmentation_span"] = aug_span
            new_example["noise"] = True
            return new_example
        else:
            return None

    def __add_noise(self, claim, aug_span):
        claim_tokens = [token.text_with_ws for token in claim]

        new_claim = []
        for ix, token in enumerate(claim_tokens):
            # don't modify text inside an augmented span
            apply_augmentation = True
            if aug_span:
                span_start, span_end = aug_span
                if span_start <= ix <= span_end:
                    apply_augmentation = False

            # decide whether to add noise
            if apply_augmentation and random.random() < self.noise_prob:
                # decide whether to replicate or delete token
                if random.random() < self.delete_prob:
                    # update spans and skip token
                    if aug_span:
                        span_start, span_end = aug_span
                        if ix < span_start:
                            span_start -= 1
                            span_end -= 1
                        aug_span = span_start, span_end
                    if len(new_claim) > 0:
                        if new_claim[-1][-1] != " ":
                            new_claim[-1] = new_claim[-1] + " "
                    continue
                else:
                    if aug_span:
                        span_start, span_end = aug_span
                        if ix < span_start:
                            span_start += 1
                            span_end += 1
                        aug_span = span_start, span_end
                    new_claim.append(token)
            new_claim.append(token)
        new_claim = self.spacy("".join(new_claim))

        if claim.text == new_claim.text:
            return None, None
        else:
            return new_claim, aug_span


#----------------------------------------------------------




class sample(Transformation):
    # Inject noise into claims
    def __init__(self):
        super().__init__()

        self.T1=PronounSwap()
        self.T2=EntitySwap()
        self.T3=NumberSwap()
        self.T4=DateSwap()
        self.T5=AddNoise()  
        self.nlp=spacy.load("en_core_web_sm")
        

    def neg_tran(self,source,tar):

        example={}
        example['text']=self.nlp(source)
        example['claim']=self.nlp(tar)
        length=len(example['claim'])
        stay_true=int(length*random.random())
        example['augmentation_span']=(0,stay_true)

        out=None
        method=0
        count=0
        while(out == None):
            p=random.random()

            if 0<p<=0.3:
                out=self.T1.transform(example)
                method=1
            elif 0.3<p<=0.5:
                out=self.T2.transform(example)
                method=2            
            elif 0.5<p<=0.7:
                out=self.T3.transform(example)
                method=3            
            elif 0.7<p<=0.9:
                out=self.T4.transform(example)
                method=4           
            else:
                out=self.T5.transform(example)
                method=5
            count=count+1
            if count>10:
                break
                
        if method == 5:
            end=min(stay_true+3,length)
        else:
            end=min(out['augmentation_span'][1]+1,length)
            
        if out != None:
            return source, str(out['claim'][:end]),method
        else:
            return 0
        
    def pos_tran(self,source,tar):
        
        example={}
        example['claim']=self.nlp(tar)
        length=len(example['claim'])
        stay_true=int(length*random.random())
    
        end=min(stay_true+1,length)
    
        return source, str(example['claim'][:end]),0
    
    
    def sample_tran(self,source,tar):
        
        p=random.random()
        try:
            if p<0.5:
                seq,cla,t=self.pos_tran(source,tar)
                label=1
            else:
                seq,cla,t=self.neg_tran(source,tar)
                label=0
            return seq,cla,label,t   
        except:
            seq,cla,t=self.pos_tran(source,tar)
            label=1
            return seq,cla,label,t            


class sample_all(Transformation):
    # Inject noise into claims
    def __init__(self):
        super().__init__()

        self.T1=PronounSwap()
        self.T2=EntitySwap()
        self.T3=NumberSwap()
        self.T4=DateSwap()
        self.T5=AddNoise()  
        self.nlp=spacy.load("en_core_web_sm")
        

    def neg_tran(self,source,tar):

        example={}
        example['text']=self.nlp(source)
        example['claim']=self.nlp(tar)
        length=len(example['claim'])
        stay_true=int(length*random.random())
        example['augmentation_span']=(0,stay_true)

        out=None
        method=0
        count=0
        while(out == None):
            p=random.random()

            if 0<p<=0.3:
                out=self.T1.transform(example)
                method=1
            elif 0.3<p<=0.5:
                out=self.T2.transform(example)
                method=2            
            elif 0.5<p<=0.7:
                out=self.T3.transform(example)
                method=3            
            elif 0.7<p<=0.9:
                out=self.T4.transform(example)
                method=4           
            else:
                out=self.T5.transform(example)
                method=5
            count=count+1
            if count>10:
                break
                
        if method == 5:
            end=min(stay_true+3,length)
        else:
            end=min(out['augmentation_span'][1]+1,length)
            
        if out != None:
            return source, str(out['claim']),method
        else:
            return 0
        
    def pos_tran(self,source,tar):
        
        example={}
        example['claim']=self.nlp(tar)
    
        return source, str(example['claim']),0
    
    
    def sample_tran(self,source,tar):
        
        p=random.random()
        try:
            if p<0.5:
                seq,cla,t=self.pos_tran(source,tar)
                label=1
            else:
                seq,cla,t=self.neg_tran(source,tar)
                label=0
            return seq,cla,label,t   
        except:
            seq,cla,t=self.pos_tran(source,tar)
            label=1
            return seq,cla,label,t   



'''

data_path='data_file/multi/train/*'
filelist = glob.glob(data_path)

count=0
for batch_path in filelist:
    
    print(count)
    f=open(batch_path,'rb')
    one_batch= pickle.load(f)
    
    article_no_res=one_batch['article']
    abstract=one_batch['abstract']
    
    
    input_seq=[]
    labels=[]
    trans=[]
    for i in range(len(article_no_res)):

        raw_source=' '
        for j in article_no_res[i]:
            raw_source=raw_source+j+' '
        raw_source=raw_source.strip()
        raw_claim=abstract[i]
        source,claim,one_label,t=sample_tran(raw_source,raw_claim)
        input_seq.append((source,claim))
        labels.append(one_label)
        trans.append(t)
        
    new_batch={}
    new_batch['input_seq']=input_seq
    new_batch['labels']=labels
    new_batch['trans']=trans

    
    
    f=open('data_fact/multi/train/'+str(count)+'_train_batch_of '+str(len(input_seq))+' examples.pkl','wb')  
    pickle.dump(new_batch,f)  
    f.close() 
    count=count+1
    
    if count%20==0:
        print('finish',count,'batches')
'''