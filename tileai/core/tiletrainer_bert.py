#from transformers import BertTokenizerFast, TFBertForSequenceClassification, TFTrainer, TFTrainingArguments, BertTokenizer

from transformers import AutoModelForSequenceClassification

import torch
from torch import nn
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support

import os
import logging
from tileai.core.abstract_tiletrainer import TileTrainer
from tileai.core.preprocessing.textprocessing import prepare_dataset_bert, save_model, load_model
from tileai.shared import const

logger = logging.getLogger(__name__)

class TileTrainertorchBert(TileTrainer):
    
    def __init__(self, language, pipeline, parameters, model):
        self.language=language
        self.pipeline=pipeline
        self.parameters=parameters
        self.model = model
        





    def train(self, train_texts,train_labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["WANDB_DISABLED"] = "true"
        
        #label_encoder = LabelEncoder()
        #label_integer_encoded = label_encoder.fit_transform(train_labels)
        #print("integer encoded ",label_integer_encoded)          
        

        
        
        #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_one_hot_encoded, test_size=.1)
        #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, label_integer_encoded, test_size=.1)
        #print("=============================================================")
        #print(train_texts,train_labels)
        #print("============================================================")
        train_texts, val_texts, train_labels, val_labels, label_encoder = prepare_dataset_bert(train_texts,train_labels)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pipeline[1], cache_dir="./models/trasformer_cache")
        


        
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        
        train_dataset = TileDataset(train_encodings, train_labels)
        val_dataset = TileDataset(val_encodings, val_labels)
        
       
        # define hyperparams
        num_labels = len(set(train_labels))
        #learninge_rate = 5e-6
        #epochs = 3
        #batch_size = 16

        from transformers import TrainingArguments

     
        """
        evaluation_strategy ='steps',
        eval_steps = 50, # Evaluation and Save happens every 50 steps
        save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=False,
        metric_for_best_model = 'f1',
        load_best_model_at_end=True
        """
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=100,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            push_to_hub=False,               #aggiunte per le metriche e l'earlystopping rif: https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
            evaluation_strategy ='steps',
            eval_steps = 50, # Evaluation and Save happens every 50 steps
            save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
            learning_rate=2e-5,
            load_best_model_at_end=True,
           
        )
        
        
        model = AutoModelForSequenceClassification.from_pretrained(self.pipeline[1], num_labels=num_labels)

        model.to(device)
        
        from transformers import Trainer

        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset, 
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics, #aggiunte per earlystopping rif: https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
			callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
        # create BERT model
        trainer.train()

       
        model.save_pretrained(self.model)
      
        
        
        id2label = {}
        label2id = {}
        for cla in label_encoder.classes_:
            id2label[str(label_encoder.transform([cla])[0])]=cla
            label2id[cla]=int(label_encoder.transform([cla])[0])
        


        configuration = {}
        configuration["language"] = self.language
        configuration["pipeline"] = self.pipeline
        
        configuration["id2label"]=id2label
        configuration["label2id"] = label2id
        
        config_label_json = self.model+"/"+const.MODEL_CONFIG
        print(config_label_json)
    
        with open(config_label_json, 'w', encoding='utf-8') as f:
            json.dump(configuration, f, ensure_ascii=False, indent=4)
        
        
        #return embed_classifier.state_dict(), configuration, vocab.get_itos(), creport
        return self.compute_metrics

    def query(self, configuration,query_text):
        

        for i in configuration:
            
            language = configuration["language"]
            id2label = configuration["id2label"]
            label2id = configuration["label2id"]
            pipeline = configuration["pipeline"]
        

        from transformers import AutoTokenizer, Trainer
        tokenizer = AutoTokenizer.from_pretrained(pipeline[1], cache_dir=const.MODEL_CACHE_TRANSFORMERS)
        
        #text_encodings = tokenizer([query_text], truncation=True, padding=True)
        
        
        model = AutoModelForSequenceClassification.from_pretrained(self.model)

        trainer = Trainer(model=model)
        
        input_ids = tokenizer.encode(query_text, add_special_tokens=True)
        
        # Create tensor, use .cuda() to transfer the tensor to GPU
        query_text_tensor = torch.tensor(input_ids).long()
        # Fake batch dimension

        query_text_tensor = query_text_tensor.unsqueeze(0)

        # Call the model and get the logits

        logits,  = model(query_text_tensor).logits
    
    
        # Remove the fake batch dimension
        logits = logits.squeeze(0)

        # The model was trained with a Log Likelyhood + Softmax combined loss, hence to extract probabilities we need a softmax on top of the logits tensor
        #proba = nn.functional.softmax(logits, dim=0)
        
        pred_prob = torch.softmax(logits,dim=0)

       

        predicted_class = torch.argmax(pred_prob).item()
        predicted_prob = pred_prob[predicted_class].item() 
        print(predicted_class, predicted_prob)

        

        #ciclo sulle classi ed ottengo le prob per ogni classee
            
        intent_r = []
        for idx,classes_to_pred in enumerate(pred_prob):
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


    def predict(self, text):
        from transformers import AutoTokenizer, Trainer
        tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-uncased', cache_dir=const.MODEL_CACHE_TRANSFORMERS)
       
        test_encodings = tokenizer([text], truncation=True, padding=True)
            

        test_dataset = SimpleDataset(test_encodings)
        
        
       
        trainer = Trainer(model=self.model)
        
        predictions = trainer.predict(test_dataset)
        print(predictions.predictions)
        

        self.model.eval()
        pt_inputs = {k: torch.tensor(v).to(trainer.args.device) for k, v in test_encodings.items()}
        with torch.no_grad():
            output = self.model(**pt_inputs)
        print(output.logits.cpu().numpy())

    def predict1(self, text):
        from transformers import AutoTokenizer, Trainer
        tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-uncased', cache_dir="./models/trasformer_cache")
       
        
        test_encodings = tokenizer([text], truncation=True, padding=True)
            

        test_dataset = SimpleDataset(test_encodings)
        
        
       
        trainer = Trainer(model=self.model)
        
        
        
        
        predictions = trainer.predict(test_dataset)
        print(predictions.predictions)
        
        
        input_ids = tokenizer.encode(text, add_special_tokens=True)

        # Create tensor, use .cuda() to transfer the tensor to GPU
        tensor = torch.tensor(input_ids).long()
        # Fake batch dimension

        tensor = tensor.unsqueeze(0)

        # Call the model and get the logits

        logits,  = self.model(tensor).logits
    
    
        # Remove the fake batch dimension
        logits = logits.squeeze(0)

        # The model was trained with a Log Likelyhood + Softmax combined loss, hence to extract probabilities we need a softmax on top of the logits tensor
        proba = nn.functional.softmax(logits, dim=0)

        # Unpack the tensor to obtain negative, neutral and positive probabilities
        print(proba)
        

        max_value = proba.argmax().item()
        print(max_value)
        #print(negative.item(), neutral.item(), positive.item())
        #saluti 3
        #chisei 0
        #dovesei 1
        #ristorante 2
        #taxi 4
        

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
    
    
    def query_http(self, vocabll=None, model_classifier=None, id2label=None, tokenizer=None, query_text=None):
       
        #trainer = Trainer(model=model_classifier)
         
        
        input_ids = tokenizer.encode(query_text, add_special_tokens=True)
        
        # Create tensor, use .cuda() to transfer the tensor to GPU
        query_text_tensor = torch.tensor(input_ids).long()
        # Fake batch dimension

        query_text_tensor = query_text_tensor.unsqueeze(0)

        # Call the model and get the logits

        logits,  = model_classifier(query_text_tensor).logits
    
    
        # Remove the fake batch dimension
        logits = logits.squeeze(0)

        # The model was trained with a Log Likelyhood + Softmax combined loss, hence to extract probabilities we need a softmax on top of the logits tensor
        #proba = nn.functional.softmax(logits, dim=0)
        
        pred_prob = torch.softmax(logits,dim=0)

       

        predicted_class = torch.argmax(pred_prob).item()
        predicted_prob = pred_prob[predicted_class].item() 
        print(predicted_class, predicted_prob)

        

        #ciclo sulle classi ed ottengo le prob per ogni classee
            
        intent_r = []
        for idx,classes_to_pred in enumerate(pred_prob):
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

