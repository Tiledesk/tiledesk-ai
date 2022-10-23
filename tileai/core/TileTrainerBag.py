import torch
from torch import nn
import torchtext
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


logger = logging.getLogger(__name__)

class TileTrainertorchBag:
    
    def __init__(self, language, algo, parameters, model):
        self.language=language
        self.algo=algo
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
        label_pipeline = lambda x: train_labels[x] #int(x) - 1
        
      
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
        LR = 5
        BATCH_SIZE = 32

        loss_fn = nn.CrossEntropyLoss() #criterion
        embed_classifier = TextClassificationModel(len(vocab), len(target_classes)).to(device)
        #optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)
        optimizer = SGD(embed_classifier.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        total_accu = None
        
        num_train = int(len(train_dataset) * 0.95)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch) #1024
        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch) #1024
        
        for epoch in range(1, EPOCHS + 1):
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
            print("Valid Loss : {:.3f}".format(lll))



               
       
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
        print("FINE")

        print(self.predict("voglio un taxi", text_pipeline, model=embed_classifier))
        
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
        configuration["class"]=type(embed_classifier).__name__
        configuration["module"]=type(embed_classifier).__module__
        configuration["id2label"]=id2label
        configuration["label2id"] = label2id
        configuration["vocab_size"]=len(vocab)
        
        return embed_classifier.state_dict(), configuration, vocab.get_itos(), creport
    
    async def trainwlstmtrain(self, train_texts,train_labels):
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = train_texts
        labelmia= train_labels
        label_encoder = LabelEncoder()
        label_integer_encoded = label_encoder.fit_transform(train_labels)

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
        
        print("======== ", len(vocab))
        
    
        train_dataset, test_dataset = to_map_style_dataset(train_texts), to_map_style_dataset(val_texts)
        target_classes = set(train_labels)
        
        def vectorize_batch(batch):
            Y, X = list(zip(*batch))
            X = [vocab(tokenizer(sample)) for sample in X]
            X = [sample+([0]* (20-len(sample))) if len(sample)<20 else sample[:20] for sample in X] ## Bringing all samples to 50 length. #50
            return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)        
        
        
        

        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: train_labels[x] #int(x) - 1
        
      
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
        LR = 5
        BATCH_SIZE = 32

        #loss_fn = nn.CrossEntropyLoss() #criterion
        loss_fn =nn.NLLLoss()
        embed_classifier = LSTMClassificationModel(vocab, target_classes).to(device)
        #optimizer = Adam(embed_classifier.parameters(), lr=LR)
        optimizer = SGD(embed_classifier.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        total_accu = None
        
        num_train = int(len(train_dataset) * 0.95)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch) #1024
        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)
        test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch) #1024
        
                  
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            self.train_lstmmodel(train_dataloader,model=embed_classifier, optimizer=optimizer, criterion = loss_fn )
            accu_val, lossss = self.evaluatelstm(valid_dataloader,model=embed_classifier, optimizer=optimizer ,criterion = loss_fn)
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
            print("Valid Loss : {:.3f}".format(lossss))
        
        



               
       
        Y_actual, Y_preds = self.makePredictionslstm(embed_classifier, test_dataloader)
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        

        from sklearn.metrics import confusion_matrix
        import scikitplot as skplt
        import matplotlib.pyplot as plt
        
        
        skplt.metrics.plot_confusion_matrix([i for i in Y_actual], [i for i in Y_preds],
            normalize=True,
            title="Confusion Matrix",
            cmap="Reds",
            hide_zeros=False,
            figsize=(5,5)
            );
        plt.xticks(rotation=90);
        plt.show()
        print(train_labels)
        print('Checking the results of test dataset.')
        accu_test,_ = self.evaluate(test_dataloader,model=embed_classifier, optimizer=optimizer ,criterion = loss_fn)
        print('test accuracy {:8.3f}'.format(accu_test))

        print("Controllo le performance")
        print("vero",Y_actual)
        print("previsto", Y_preds)
        print("FINE")

        print(self.predict("voglio un taxi", text_pipeline, model=embed_classifier))
        
        print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_actual, Y_preds))
        print("\nClassification Report : ")
        print(classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds)))#target_names=target_classes,
       

        return "ok"


    
    def train_lstmmodel(self, dataloader, **kwargs):
        
        model=kwargs['model']
        optimizer= kwargs['optimizer']
        criterion = kwargs ['criterion']
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (text, label) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text)
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
    
    def evaluatelstm(self,dataloader, **kwargs ):
        model=kwargs['model']
        optimizer= kwargs['optimizer']
        criterion = kwargs ['criterion']
        model.eval()
        total_acc, total_count = 0, 0
        losses = []
        with torch.no_grad():
            for idx, (text, label) in enumerate(dataloader):
                predicted_label = model(text)
                loss = criterion(predicted_label, label)
                losses.append(loss.item())
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count, torch.tensor(losses).mean()

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
        log_interval = 500
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
        losses = []
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                losses.append(loss.item())
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count, torch.tensor(losses).mean()
    
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
    

