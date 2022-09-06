import torch
from torch import nn
import torchtext
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
from torch.optim import Adam
from tileai.classifier.torch_classifiers import EmbeddingClassifier,EmbeddingClassifierAverage, EmbeddingClassifierWBag, TextClassificationModel
from torch.nn import functional as F
import time


class TileTrainertorchFF:
    
    def __init__(self, language, modelname, parameters, model):
        self.language=language
        self.modelname=modelname
        self.parameters=parameters
        self.model = model
        

    
    def train(self, train_texts,train_labels):
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = train_texts

        label_encoder = LabelEncoder()
        label_integer_encoded = label_encoder.fit_transform(train_labels)
        print("integer encoded ",label_integer_encoded)          
        # one hot encode labels


        onehot_encoder = OneHotEncoder(sparse=False)
        label_integer_encoded_reshaped = label_integer_encoded.reshape(len(label_integer_encoded), 1)
        label_one_hot_encoded = onehot_encoder.fit_transform(label_integer_encoded_reshaped)
        
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
        
        
        """
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
        """

        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        test_loader  = DataLoader(test_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        
        epochs = 200
        learning_rate = 5e-4

        loss_fn = nn.CrossEntropyLoss()
        embed_classifier = EmbeddingClassifier(vocab, target_classes)
        optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)

        self.trainModel(embed_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)

        
        Y_actual, Y_preds = self.makePredictions(embed_classifier, test_loader)
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        
       

        from sklearn.metrics import confusion_matrix
        import scikitplot as skplt
        import matplotlib.pyplot as plt
        

        skplt.metrics.plot_confusion_matrix([i for i in Y_actual], [i for i in Y_preds],
            normalize=True,
            title="Confusion Matrix",
            cmap="Reds",
            hide_zeros=True,
            figsize=(5,5)
            );
        plt.xticks(rotation=90);
        plt.show()

        print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_actual, Y_preds))
        print("\nClassification Report : ")
        print(classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds)))#target_names=target_classes,

        # Print model's state_dict
        #print("Model's state_dict:")
        #for param_tensor in embed_classifier.state_dict():
        #    print(param_tensor, "\t", embed_classifier.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        #print("Optimizer's state_dict:")
        #for var_name in optimizer.state_dict():
        #    print(var_name, "\t", optimizer.state_dict()[var_name])

        
        return embed_classifier.state_dict()


    async def trainaverage(self, train_texts,train_labels):
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = train_texts

        label_encoder = LabelEncoder()
        label_integer_encoded = label_encoder.fit_transform(train_labels)
        print("integer encoded ",label_integer_encoded)          
        # one hot encode labels


        onehot_encoder = OneHotEncoder(sparse=False)
        label_integer_encoded_reshaped = label_integer_encoded.reshape(len(label_integer_encoded), 1)
        label_one_hot_encoded = onehot_encoder.fit_transform(label_integer_encoded_reshaped)
        
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
        
        
        """
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
        """

        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        test_loader  = DataLoader(test_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        
        epochs = 200
        learning_rate = 5e-4

        loss_fn = nn.CrossEntropyLoss()
        embed_classifier = EmbeddingClassifierAverage(vocab, target_classes)
        optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)

        self.trainModel(embed_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)

        
        Y_actual, Y_preds = self.makePredictions(embed_classifier, test_loader)
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

       
       

        from sklearn.metrics import confusion_matrix
        import scikitplot as skplt
        import matplotlib.pyplot as plt
        

        skplt.metrics.plot_confusion_matrix([i for i in Y_actual], [i for i in Y_preds],
            normalize=True,
            title="Confusion Matrix",
            cmap="Reds",
            hide_zeros=True,
            figsize=(5,5)
            );
        plt.xticks(rotation=90);
        plt.show()

        print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_actual, Y_preds))
        print("\nClassification Report : ")
        print(classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds)))#target_names=target_classes,


        return "ok"

    async def trainwbag(self, train_texts,train_labels):
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = train_texts

        label_encoder = LabelEncoder()
        label_integer_encoded = label_encoder.fit_transform(train_labels)
        print("integer encoded ",label_integer_encoded)          
        # one hot encode labels


        #onehot_encoder = OneHotEncoder(sparse=False)
        #label_integer_encoded_reshaped = label_integer_encoded.reshape(len(label_integer_encoded), 1)
        #label_one_hot_encoded = onehot_encoder.fit_transform(label_integer_encoded_reshaped)
        
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
        
        
        """
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
        """

        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        test_loader  = DataLoader(test_dataset, batch_size=32, collate_fn=vectorize_batch) #1024
        
        epochs = 200
        learning_rate = 5e-4

        loss_fn = nn.CrossEntropyLoss()
        embed_classifier = EmbeddingClassifierWBag(vocab, target_classes)
        optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)

        self.trainModel(embed_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)

        
        Y_actual, Y_preds = self.makePredictions(embed_classifier, test_loader)
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        
       

        from sklearn.metrics import confusion_matrix
        import scikitplot as skplt
        import matplotlib.pyplot as plt
        

        skplt.metrics.plot_confusion_matrix([i for i in Y_actual], [i for i in Y_preds],
            normalize=True,
            title="Confusion Matrix",
            cmap="Reds",
            hide_zeros=True,
            figsize=(5,5)
            );
        plt.xticks(rotation=90);
        plt.show()


        print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(Y_actual, Y_preds))
        print("\nClassification Report : ")
        print(classification_report(Y_actual, Y_preds,  labels=np.unique(Y_preds)))#target_names=target_classes,

        return "ok"
    
    



    def tokenizer(self, inp_str): ## This method is one way of creating tokenizer that looks for word tokens
        return re.findall(r"\w+", inp_str)

    def build_vocab(self,datasets):
        tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
        for dataset in datasets:
            yield tokenizer(dataset)

    

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


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
		
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

