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

from tileai.core.classifier.torch_classifiers import EmbeddingClassifierAverage
from tileai.core.abstract_tiletrainer import TileTrainer
from tileai.shared import const
from tileai.core.preprocessing.textprocessing import prepare_dataset, save_model, load_model
from tileai.core.tokenizer.standard_tokenizer import StandarTokenizer


logger = logging.getLogger(__name__)

#EmbeddingClassifierAverage
class TileTrainertorchAverage(TileTrainer):

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

        train_texts, val_texts, train_labels, val_labels, label_encoder = prepare_dataset(train_texts,train_labels)

        tokenizer = StandarTokenizer()#get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
        vocabulary = self.build_vocab(dataset)
        

        vocab = build_vocab_from_iterator(vocabulary, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        
        train_dataset, test_dataset = to_map_style_dataset(train_texts), to_map_style_dataset(val_texts)
        target_classes = set(train_labels)
        
        def vectorize_batch(batch):
            Y, X = list(zip(*batch))
            X = [vocab(tokenizer.tokenize(sample)) for sample in X]
            X = [sample+([0]* (20-len(sample))) if len(sample)<20 else sample[:20] for sample in X] ## Bringing all samples to 50 length. #50
            return torch.tensor(X, dtype=torch.int32).to(device), torch.tensor(Y).to(device)        
        
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        test_loader  = DataLoader(test_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        
        epochs = 200
        learning_rate = 5e-4

        loss_fn = nn.CrossEntropyLoss()

        
        embed_classifier = EmbeddingClassifierAverage(len(vocab), len(target_classes)).to(device)
        
        optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)

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
       
        
        save_model(label_encoder,self.language, self.pipeline, embed_classifier, vocab, self.model)
        
        return creport

    def build_vocab(self,datasets):
        tokenizer = StandarTokenizer()# get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
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
            for X, Y in val_loader:
                preds = model(X)
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
            for X, Y in tqdm(train_loader):
                Y_preds = model(X)

                loss = loss_fn(Y_preds, Y)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
            self.calcValLossAndAccuracy(model, loss_fn, val_loader)
    
    def makePredictions(self, model, loader):
        Y_shuffled, Y_preds = [], []
        for X, Y in loader:
            preds = model(X)
            Y_preds.append(preds)
            Y_shuffled.append(Y)
        gc.collect()
        Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

        return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()


    def query(self, configuration, query_text):
        
        vocabll,model_classifier, id2label  = load_model(configuration, self.model)
        
        tokenizer = StandarTokenizer()# get_tokenizer("basic_english") 

          
        text_pipeline = lambda x: [vocabll[token] for token in tokenizer.tokenize(x)]

        with torch.no_grad():
            vect = [text_pipeline(query_text)]
            #text = torch.tensor([sample+([0]* (20-len(sample))) if len(sample)<20 else sample[:20] for sample in vect])
            
            text = torch.tensor(vect)
            logits_output = model_classifier(text)
            
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

