import logging

import importlib
import json
import torch

from collections import OrderedDict
from torchtext.vocab import Vocab, vocab

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tileai.shared import const

logger = logging.getLogger(__name__)

def prepare_dataset(train_texts,train_labels):
    dataset = train_texts

    label_encoder = LabelEncoder()
    label_integer_encoded = label_encoder.fit_transform(train_labels)
        
    logger.info("integer encoded ",label_integer_encoded)    
                      

        #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_one_hot_encoded, test_size=.1)
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_integer_encoded, test_size=.2)#, stratify=label_integer_encoded)
    except ValueError as ve:
        logger.info(ve)
        print(ve)
        
        val_texts = train_texts
        train_labels = label_integer_encoded
        val_labels = label_integer_encoded
        
        
    print("=============================================================")
    print(train_texts,train_labels)
    print("============================================================")
        
        
    train_texts = zip(train_labels,train_texts)
        
        
    val_texts = zip(val_labels, val_texts)

    return train_texts, val_texts, train_labels, val_labels, label_encoder

def prepare_dataset_bert(train_texts,train_labels):
    dataset = train_texts

    label_encoder = LabelEncoder()
    label_integer_encoded = label_encoder.fit_transform(train_labels)
        
    logger.info("integer encoded ",label_integer_encoded)    
                      

        #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_one_hot_encoded, test_size=.1)
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_integer_encoded, test_size=.2)#, stratify=label_integer_encoded)
    except ValueError as ve:
        logger.info(ve)
        print(ve)
        
        val_texts = train_texts
        train_labels = label_integer_encoded
        val_labels = label_integer_encoded
        
        
    print("=============================================================")
    print(train_texts,train_labels)
    print("============================================================")
        
        
   
    return train_texts, val_texts, train_labels, val_labels, label_encoder

def save_model(label_encoder, language, pipeline, embed_classifier, vocab, model):

    id2label = {}
    label2id = {}
    for cla in label_encoder.classes_:
        id2label[str(label_encoder.transform([cla])[0])]=cla
        label2id[cla]=int(label_encoder.transform([cla])[0])
        
    configuration = {}
    configuration["language"] = language
    configuration["pipeline"] = pipeline
    configuration["class"]=type(embed_classifier).__name__
    configuration["module"]=type(embed_classifier).__module__
    configuration["id2label"]=id2label
    configuration["label2id"] = label2id
    configuration["vocab_size"]=len(vocab)
       
    torch.save (embed_classifier.state_dict(), model+"/"+const.MODEL_BIN)

    config_json = model+"/"+const.MODEL_CONFIG
    vocab_file = model+"/"+const.MODEL_VOC
    print(config_json)
    
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(configuration, f, ensure_ascii=False, indent=4)
    
    f.close()
    print(vocab)
    with open(vocab_file, 'w', encoding='utf-8') as f_v:
        for vb in vocab.get_itos():
            f_v.write(vb)
            f_v.write("\n")
        
    f_v.close()  

def load_model(configuration, model):
    
    
    vocabulary = []
    vocab_file = model+"/"+const.MODEL_VOC
    vocabulary = open (vocab_file, "r",  encoding='utf-8').read().splitlines()
    
    model_file = model+"/"+const.MODEL_BIN

    for i in configuration:
        language = configuration["language"]
        embed_class = configuration["class"]
        embed_module = configuration["module"]
        id2label = configuration["id2label"]
        label2id = configuration["label2id"]
        vocab_size = configuration["vocab_size"]

        
    module = importlib.import_module(embed_module)
    class_ = getattr(module, embed_class)
        
    model_classifier = class_(vocab_size, len(id2label.keys()))
    model_classifier.load_state_dict(torch.load(model_file))
    model_classifier.eval()
 
    odict = OrderedDict([(v,1) for v in vocabulary])
        
    vocab_for_query = vocab(odict, specials=["<unk>"])
    vocab_for_query.set_default_index(vocab_for_query["<unk>"])
    vocabll = Vocab(vocab_for_query)
    
    return vocabll, model_classifier, id2label

def load_model_bert(configuration, model):
    #model_classifier, id2label, tokenizer
    from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(configuration["pipeline"][1], cache_dir=const.MODEL_CACHE_TRANSFORMERS)
    
    config_file_path = model+"/config.json"
    config_json = open (config_file_path, "r",  encoding='utf-8').read().splitlines()
    
    model_file = model+"/pytorch_model.bin"

    for i in configuration:
        language = configuration["language"]
       
        id2label = configuration["id2label"]
        label2id = configuration["label2id"]
       
        
    model = AutoModelForSequenceClassification.from_pretrained(model)

   
    
    
    return model, id2label, tokenizer
    
    