import json

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import torch

from tileai.shared import const
from tileai.core.abstract_tiletrainer import TileTrainer
from tileai.core.preprocessing.data_reader import make_dataframe
from tileai.core.preprocessing.diet_dataset import DIETClassifierDataset
from tileai.core.classifier.diet_classifier import DIETClassifier, DIETClassifierConfig
from tileai.core.classifier.diet_trainer import DIETTrainer
from tileai.core.classifier.diet_wrapper import DIETClassifierWrapper
from transformers import AutoTokenizer




class TileTrainertorchDIET(TileTrainer):




    def __init__(self, language, pipeline, parameters, model):
        self.language=language
        self.pipeline=pipeline
        self.parameters=parameters
        self.model = model
    
    def train(self, dataframe,entities_list, intents_list, synonym_dict):


        wrapper = DIETClassifierWrapper(
            config={"language":self.language,
                    "pipeline":self.pipeline,
                    "parameters":self.parameters,
                    "model":self.model
        },entities_list=entities_list, intents_list=intents_list, synonym_dict=synonym_dict)
        wrapper.train_model(dataframe=dataframe, synonym_dict=synonym_dict)

        
       
        #id2label = {}
        #label2id = {}
        #for cla in label_encoder.classes_:
        #    id2label[str(label_encoder.transform([cla])[0])]=cla
        #    label2id[cla]=int(label_encoder.transform([cla])[0])
        


        configuration = {}
        configuration["language"] = self.language
        configuration["pipeline"] = self.pipeline
        
        configuration["entities"] = entities_list
        configuration["intents"] = intents_list
        configuration["synonim"] = synonym_dict
        
        config_label_json = self.model+"/"+const.MODEL_CONFIG
        print(config_label_json)
    
        with open(config_label_json, 'w', encoding='utf-8') as f:
            json.dump(configuration, f, ensure_ascii=False, indent=4)
        
        
        #return embed_classifier.state_dict(), configuration, vocab.get_itos(), creport
        return self.compute_metrics
        
        
        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        #tokenizer = AutoTokenizer.from_pretrained(self.pipeline[1], cache_dir=const.MODEL_CACHE_TRANSFORMERS)

        
        #entities_list = [entity for entity in entities_list if entity != "number"]
        #print(f"ENTITIES_LIST: {entities_list}")
        
        #dataset = DIETClassifierDataset(dataframe=dataframe, tokenizer=tokenizer, entities=entities_list, intents=intents_list)

        #config = DIETClassifierConfig(model=self.pipeline[1], entities=entities_list, intents=intents_list)
        #model = DIETClassifier(config=config)

        #sentences = ["What if I'm late"]

        #inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", max_length=512)

        #outputs = model(**{k: v for k, v in inputs.items()})

        #trainer = DIETTrainer(model=model, dataset=dataset, train_range=0.95, num_train_epochs=1)

        #trainer.train()

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

    def query(self, configuration, query_text):
        pass

    def query_http(self, vocabll,model_classifier, id2label,tokenizer, query_text):
        pass