import torch
from torch import nn
import torchtext
from collections import OrderedDict
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab, vocab
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc
from torch.optim import Adam, SGD
from tileai.core.classifier.torch_classifiers import TextClassificationModel, LSTMClassificationModel
from torch.nn import functional as F
import time

import importlib
import logging
from tileai.core.abstract_tiletrainer import TileTrainer


logger = logging.getLogger(__name__)

class TileTrainertorchBag(TileTrainer):
    
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
       
        tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
        vocabulary = self.build_vocab(dataset)

        vocab = build_vocab_from_iterator(vocabulary, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
            
    
        train_dataset, test_dataset = to_map_style_dataset(train_texts), to_map_style_dataset(val_texts)
        target_classes = set(train_labels)
        
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: train_labels[x] 
      
        def collate_batch(batch):
            label_list, text_list, offsets = [], [], [0]
            for (_label, _text) in batch:
                label_list.append(label_pipeline(_label))
                
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))
            label_list = torch.tensor(label_list, dtype=torch.int64)
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text_list = torch.cat(text_list)
            return label_list.to(device), text_list.to(device), offsets.to(device) 
    
        
        EPOCHS = 200
        learning_rate = 5e-4
        LR = 5e-4#5
        BATCH_SIZE = 32

        loss_fn = nn.CrossEntropyLoss() #criterion
        embed_classifier = TextClassificationModel(len(vocab), len(target_classes)).to(device)
        optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)
        #optimizer = SGD(embed_classifier.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        total_accu = None
        
        num_train = int(len(train_dataset) * 0.95)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch) #1024
        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch) #1024
        
        
        for epoch in range(1, EPOCHS + 1):
            losses = []
            epoch_start_time = time.time()
            self.trainwtexttrain(train_dataloader,model=embed_classifier, optimizer=optimizer, criterion = loss_fn )
            accu_val, lll = self.evaluate(valid_dataloader,model=embed_classifier, optimizer=optimizer ,criterion = loss_fn)
            if total_accu is not None and total_accu > accu_val:
                scheduler.step()
            else:
                total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f} '.format(epoch,
                                                   time.time() - epoch_start_time,
                                                   accu_val))
            print('-' * 59)
            
            losses.append(lll)
            print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))



               
       
        Y_actual, Y_preds = self.makePredictions(embed_classifier, test_dataloader)
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        

        from sklearn.metrics import confusion_matrix
      
        print(train_labels)
        print('Checking the results of test dataset.')
        accu_test, _ = self.evaluate(test_dataloader,model=embed_classifier, optimizer=optimizer ,criterion = loss_fn)
        print('test accuracy {:8.3f}'.format(accu_test))

        print("Controllo le performance")
        print("vero",Y_actual)
        print("previsto", Y_preds)
        
        print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_actual, Y_preds))
        print("\nClassification Report : ")
        creport = classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds), output_dict=True)
        print(classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds)))#target_names=target_classes,
       

        id2label = {}
        label2id = {}
        for cla in label_encoder.classes_:
            id2label[str(label_encoder.transform([cla])[0])]=cla
            label2id[cla]=int(label_encoder.transform([cla])[0])
        


        configuration = {}
        configuration["pipeline"] = self.pipeline
        configuration["class"]=type(embed_classifier).__name__
        configuration["module"]=type(embed_classifier).__module__
        configuration["id2label"]=id2label
        configuration["label2id"] = label2id
        configuration["vocab_size"]=len(vocab)
        
        return embed_classifier.state_dict(), configuration, vocab.get_itos(), creport
    
   

    def validation_metrics (self, model, valid_dl):
        model.eval()
        correct = 0
        total = 0
        sum_loss = 0.0
        sum_rmse = 0.0
        for x, y, l in valid_dl:
            x = x.long()
            y = y.long()
            y_hat = model(x, l)
            #loss = F.cross_entropy(y_hat, y)
            loss = criterion(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
        return sum_loss/total, correct/total, sum_rmse/total



    def trainwtexttrain(self,dataloader, **kwargs ):
        model=kwargs['model']
        optimizer= kwargs['optimizer']
        criterion = kwargs ['criterion']
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 5
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                   total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(self,dataloader, **kwargs ):
        model=kwargs['model']
        optimizer= kwargs['optimizer']
        criterion = kwargs ['criterion']
        model.eval()
        total_acc, total_count = 0, 0

        losses_ =[]
        
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                losses_.append(loss.item())
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count, losses_
    
    def tokenizer(self, inp_str): ## This method is one way of creating tokenizer that looks for word tokens
        return re.findall(r"\w+", inp_str)

    def build_vocab(self,datasets):
        tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
        for dataset in datasets:
            yield tokenizer(dataset)
    
    def predict(self, text, text_pipeline, **kwargs):
        model=kwargs['model']
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() 
    
    
    def makePredictions(self, model, loader):
        Y_shuffled, Y_preds = [], []
        for Y, X, offset in loader:
            preds = model(X,offset)
            Y_preds.append(preds)
            Y_shuffled.append(Y)
        gc.collect()
        Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

        return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()
    
    def makePredictionslstm(self, model, loader):
        Y_shuffled, Y_preds = [], []
        for X, Y in loader:
            preds = model(X)
            Y_preds.append(preds)
            Y_shuffled.append(Y)
        gc.collect()
        Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

        return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()
    
    def query(self, model_file,configuration, vocabulary,query_text):
        
        
        for i in configuration:
            
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
        tokenizer = get_tokenizer("basic_english") 

          
        #text_pipeline = lambda x: [vocabll[token] for token in tokenizer(x)]
        text_pipeline = lambda x: vocabll(tokenizer(x))

        with torch.no_grad():
            #vect = [text_pipeline(query_text)]
            #text = torch.tensor([sample+([0]* (20-len(sample))) if len(sample)<20 else sample[:20] for sample in vect])
            #text = torch.tensor(vect)
            #prova = text_pipeline(query_text)
            #print(prova)
            text = torch.tensor(text_pipeline(query_text))
            

            
            logits_output = model_classifier(text,torch.tensor([0]) )
            
            print(logits_output.argmax(1).item() )
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
            

       
        return id2label[str(predicted_class)], model_classifier, vocabll, results_dict

