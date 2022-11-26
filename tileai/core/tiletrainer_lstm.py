import json
import numpy as np
import re
from tqdm import tqdm
import gc
from collections import OrderedDict
import time
import importlib
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support

import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab, vocab
from torchtext.data.functional import to_map_style_dataset

from tileai.core.classifier.torch_classifiers import LSTMClassificationModel
from tileai.core.tokenizer.standard_tokenizer import StandarTokenizer

from tileai.core.abstract_tiletrainer import TileTrainer
from tileai.shared import const

logger = logging.getLogger(__name__)

#EmbeddingClassifier 

class TileTrainertorchLSTM(TileTrainer):

    """
    
    """    

    def __init__(self, language, pipeline, parameters, model):
        self.language=language
        self.pipeline=pipeline
        self.parameters=parameters
        self.model = model
          
    def train(self, train_texts,train_labels):
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = train_texts

        label_encoder = LabelEncoder()
        label_integer_encoded = label_encoder.fit_transform(train_labels)
        
        logger.info("integer encoded ",label_integer_encoded)                  

        #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_one_hot_encoded, test_size=.1)
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_integer_encoded, test_size=.2)
        print("=============================================================")
        print(train_texts,train_labels)
        print("============================================================")

       
        train_texts = zip(train_labels,train_texts )
        
        
        val_texts =zip(val_labels, val_texts)
    

        #tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
        tokenizer = StandarTokenizer()
        vocabulary = self.build_vocab(dataset)
        

        vocab = build_vocab_from_iterator(vocabulary, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        
        train_dataset, test_dataset = to_map_style_dataset(train_texts), to_map_style_dataset(val_texts)
        target_classes = set(train_labels)
        
        def vectorize_batch(batch):
            Y, X = list(zip(*batch))
            X = [vocab(tokenizer.tokenize(sample)) for sample in X]
            
            X = [sample+([0]* (20-len(sample))) if len(sample)<20 else sample[:20] for sample in X] ## Bringing all samples to 20 length. #50
            lenx = [len(sample) for sample in X]
            
            
            return torch.tensor(X, dtype=torch.int32).to(device), torch.tensor(Y).to(device),torch.tensor(lenx, dtype=torch.int64).to(device)       
        
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        test_loader  = DataLoader(test_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        
        epochs = 200
        learning_rate = 5e-4

        loss_fn = nn.CrossEntropyLoss()

        embed_classifier = LSTMClassificationModel(len(vocab), len(target_classes)).to(device)

        optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)
        for x,y,z in train_loader:
            print(x,y,z)

        self.trainModel(embed_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)

        
        Y_actual, Y_preds = self.makePredictions(embed_classifier, test_loader)
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        
       

        from sklearn.metrics import confusion_matrix
        print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_actual, Y_preds))
        print("\nClassification Report : ")
        creport = classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds), output_dict=True)
        print(classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds)))
       
        
        id2label = {}
        label2id = {}
        for cla in label_encoder.classes_:
            id2label[str(label_encoder.transform([cla])[0])]=cla
            label2id[cla]=int(label_encoder.transform([cla])[0])
        


        configuration = {}
        configuration["language"] = self.language
        configuration["pipeline"] = self.pipeline
        configuration["class"]=type(embed_classifier).__name__
        configuration["module"]=type(embed_classifier).__module__
        configuration["id2label"]=id2label
        configuration["label2id"] = label2id
        configuration["vocab_size"]=len(vocab)
        
        torch.save (embed_classifier.state_dict(), self.model+"/"+const.MODEL_BIN)

        config_json = self.model+"/"+const.MODEL_CONFIG
        vocab_file = self.model+"/"+const.MODEL_VOC
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
        
        return creport
    

    def build_vocab(self,datasets):
        tokenizer = StandarTokenizer()#get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
        for dataset in datasets:
            yield tokenizer.tokenize(dataset)

    

    # metodo per calcolare la metrica e usare earlystopping
    def compute_metrics(self,pred):
       labels = pred.label_ids
       preds = pred.predictions.argmax(-1)
       precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
       acc = accuracy_score(labels, preds)
       return {
           'accuracy': acc,
           'f1': f1,
           'precision': precision,
           'recall': recall
            }


    def calcValLossAndAccuracy(self,model, loss_fn, val_loader):
        with torch.no_grad():
            Y_shuffled, Y_preds, losses = [],[],[]
            for X, Y, lenx in val_loader:
                preds = model(X,lenx)
                loss = loss_fn(preds, Y)
                losses.append(loss.item())

                Y_shuffled.append(Y)
                Y_preds.append(preds.argmax(dim=-1))

            Y_shuffled = torch.cat(Y_shuffled)
            Y_preds = torch.cat(Y_preds)

            print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
            print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))


    def trainModel(self, model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
        for i in range(1, epochs+1):
            losses = []
          
            for X, Y, LenX in tqdm(train_loader):
                
                Y_preds = model(X,LenX)

                loss = loss_fn(Y_preds, Y)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
            self.calcValLossAndAccuracy(model, loss_fn, val_loader)
    
    def makePredictions(self, model, loader):
        Y_shuffled, Y_preds = [], []
        for X, Y, lenx in loader:
            preds = model(X,lenx)
            Y_preds.append(preds)
            Y_shuffled.append(Y)
        gc.collect()
        Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

        return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()


    def query(self, configuration, query_text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocabulary = []
        vocab_file = self.model+"/"+const.MODEL_VOC
        vocabulary = open (vocab_file, "r",  encoding='utf-8').read().splitlines()
    
        model_file =   self.model+"/"+const.MODEL_BIN

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

        #print(vocabll.get_itos())
        tokenizer = StandarTokenizer()#get_tokenizer("basic_english") 

          
        text_pipeline = lambda x: [vocabll[token] for token in tokenizer.tokenize(x)]
        
        with torch.no_grad():
            vect = [text_pipeline(query_text)]
            
            ### padding o tronco se il testo è troppo lungo
            text = torch.tensor([sample+([0]* (20-len(sample))) if len(sample)<20 else sample[:20] for sample in vect]).to(device)

            lenx = torch.tensor([len(vect)], dtype=torch.int64).to(device)  
            
            logits_output = model_classifier(text,lenx)
            
            pred_prob = torch.softmax(logits_output,dim=1)
           
            
            
            predicted_class = torch.argmax(pred_prob[0]).item()
            predicted_prob = pred_prob[0][predicted_class].item() 
            print(predicted_class, predicted_prob)
            
            #ciclo sulle classi ed ottengo le prob per ogni classee
            
            intent_r = []
            for idx,classes_to_pred in enumerate(pred_prob[0]):
                intent_r.append({"name":id2label[str(idx)],
                                     "confidence": classes_to_pred.item() }) 
                #print(classes_to_pred.item())

            #predicted_class = output.argmax(1).item() 
            results_dict = {}
            results_dict["text"]= query_text
            results_dict["intent"]={"name":id2label[str(predicted_class)], 
                                     "confidence": predicted_prob }
            
            results_dict["intent_ranking"] = sorted(intent_r, key=lambda d: d['confidence'], reverse=True) 
            

       
        return id2label[str(predicted_class)],  results_dict



