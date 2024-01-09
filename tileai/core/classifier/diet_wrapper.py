from os import path, listdir
from typing import Union, Dict, List, Any, Tuple

import torch
import json
from transformers import BertTokenizerFast

import os
import sys

sys.path.append(os.getcwd())

from tileai.core.classifier.diet_classifier import DIETClassifier, DIETClassifierConfig
from tileai.core.classifier.diet_trainer import DIETTrainer
from tileai.core.preprocessing.diet_dataset import DIETClassifierDataset
from tileai.core.preprocessing.data_reader import make_dataframe, read_from_json


class DIETClassifierWrapper:
    """Wrapper for DIETClassifier."""
    def __init__(self, config: Union[Dict[str, Dict[str, Any]], str], 
                 entities_list=None, intents_list=None,synonym_dict=None):
        """
        Create wrapper with configuration.

        :param config: config in dictionary format or path to config file (.json)
        """
        #if isinstance(config, str):
        #    try:
        #        f = open(config, "r")
        #    except Exception as ex:
        #        raise RuntimeError(f"Cannot read config file from {config}: {ex}")
        #    self.config_file_path = config
        #    config = json.load(f)

        self.config = config
        self.util_config = config.get("util", {
            "intent_threshold": 0.7,
            "entities_threshold": 1e-05,
            "ambiguous_threshold": 0.2
      })

        #if config.get("model")
        model_name = config.get("pipeline", None)[1]
        
        if not model_name:
            raise ValueError(f"Config file should have 'model' attribute")

        #self.dataset_config = model_config_dict

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #model_config_attributes = ["model", "intents", "entities"]
        # model_config_dict = {k: v for k, v in model_config_dict.items() if k in model_config_attributes}

        self.intents = intents_list
        self.entities = ["O"] + entities_list

        self.model_config = DIETClassifierConfig(model=model_name, intents=intents_list, entities=entities_list)
        #**{k: v for k, v in model_config_dict.items() if k in model_config_attributes})

        #training_config_dict = config.get("training", None)
        #if not training_config_dict:
        #    raise ValueError(f"Config file should have 'training' attribute")

        #self.training_config = training_config_dict
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
       
        self.model = DIETClassifier(config=self.model_config)

        self.model.to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.synonym_dict = {} if synonym_dict is None else synonym_dict
    
    def old__init__(self, config: Union[Dict[str, Dict[str, Any]], str]):
        """
        Create wrapper with configuration.

        :param config: config in dictionary format or path to config file (.yml)
        """
        if isinstance(config, str):
            try:
                f = open(config, "r")
            except Exception as ex:
                raise RuntimeError(f"Cannot read config file from {config}: {ex}")
            self.config_file_path = config
            config = json.load(f)

        self.config = config
        self.util_config = config.get("util", None)

        model_config_dict = config.get("model", None)
        if not model_config_dict:
            raise ValueError(f"Config file should have 'model' attribute")

        self.dataset_config = model_config_dict

        if model_config_dict["device"] is not None:
            self.device = torch.device(model_config_dict["device"]) if torch.cuda.is_available() else torch.device(
                "cpu")

        model_config_attributes = ["model", "intents", "entities"]
        # model_config_dict = {k: v for k, v in model_config_dict.items() if k in model_config_attributes}

        self.intents = model_config_dict["intents"]
        self.entities = ["O"] + model_config_dict["entities"]

        self.model_config = DIETClassifierConfig(**{k: v for k, v in model_config_dict.items() if k in model_config_attributes})

        training_config_dict = config.get("training", None)
        if not training_config_dict:
            raise ValueError(f"Config file should have 'training' attribute")

        self.training_config = training_config_dict
        self.tokenizer = BertTokenizerFast.from_pretrained(model_config_dict["tokenizer"])
        
        self.model = DIETClassifier(config=self.model_config)

        self.model.to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.synonym_dict = {} if not model_config_dict.get("synonym") else model_config_dict["synonym"]

    def tokenize(self, sentences) -> Tuple[Dict[str, Any], List[List[Tuple[int, int]]]]:
        """
        Tokenize sentences using tokenizer.
        :param sentences: list of sentences
        :return: tuple(tokenized sentences, offset_mapping for sentences)
        """
        inputs = self.tokenizer(sentences, return_tensors="pt", return_attention_mask=True, return_token_type_ids=True,
                                return_offsets_mapping=True,
                                padding=True, truncation=True)

        offset_mapping = inputs["offset_mapping"]
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "offset_mapping"}

        return inputs, offset_mapping

    def convert_intent_logits(self, intent_logits: torch.tensor) -> List[Dict[str, float]]:
        """
        Convert logits from model to predicted intent,

        :param intent_logits: output from model
        :return: dictionary of predicted intent
        """
        softmax_intents = self.softmax(intent_logits)

        predicted_intents = []

        for sentence in softmax_intents:
            sentence = sentence[0]

            sorted_sentence = sentence.clone()
            sorted_sentence, _ = torch.sort(sorted_sentence)

            if sorted_sentence[-1] >= self.util_config["intent_threshold"] and (
                    sorted_sentence[-1] - sorted_sentence[-2]) >= self.util_config["ambiguous_threshold"]:
                max_probability = torch.argmax(sentence)
                max_confidence = sentence[max_probability].item()
                
            else:
                max_probability = -1
                max_confidence = 0

            predicted_intents.append({
                "intent": None if max_probability == -1 else 
                    {"name":self.intents[max_probability],"confidence":max_confidence},
                "intent_ranking": sorted([{
                    "name":intent_name, "confidence": probability.item()} 
                     for intent_name, probability in zip(self.intents, sentence)
                ], key=lambda d: d['confidence'], reverse=True)
            })

        return predicted_intents

    def convert_entities_logits(self, entities_logits: torch.tensor, offset_mapping: torch.tensor) -> List[
        List[Dict[str, Any]]]:
        """
        Convert logits to predicted entities

        :param entities_logits: entities logits from model
        :param offset_mapping: offset mapping for sentences
        :return: list of predicted entities
        """
        softmax_entities = self.softmax(entities_logits)

        predicted_entities = []

        for sentence, offset in zip(softmax_entities, offset_mapping):
            predicted_entities.append([])
            latest_entity = None
            for word, token_offset in zip(sentence, offset[1:]):
                max_probability = torch.argmax(word)
                if word[max_probability] >= self.util_config["entities_threshold"] and max_probability != 0:
                    if self.entities[max_probability] != latest_entity:
                        latest_entity = self.entities[max_probability]
                        predicted_entities[-1].append({
                            "entity_name": self.entities[max_probability],
                            "start": token_offset[0].item(),
                            "end": token_offset[1].item()
                        })
                    else:
                        predicted_entities[-1][-1]["end"] = token_offset[1].item()
                else:
                    latest_entity = None

        return predicted_entities

    def predict(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intent and entities from sentences.

        :param sentences: list of sentences
        :return: list of prediction
        """
        inputs, offset_mapping = self.tokenize(sentences=sentences)
        outputs = self.model(**inputs)
        logits = outputs["logits"]
        
        predicted_intents = self.convert_intent_logits(intent_logits=logits[1])
        predicted_entities = self.convert_entities_logits(entities_logits=logits[0], offset_mapping=offset_mapping)
        
        
        predicted_outputs = []
        for sentence, intent_sentence, entities_sentence in zip(sentences, predicted_intents, predicted_entities):
            predicted_outputs.append({})
            predicted_outputs[-1]["text"] = sentence
            predicted_outputs[-1].update(intent_sentence)
            predicted_outputs[-1].update({"entities": entities_sentence})
            for entity in predicted_outputs[-1]["entities"]:
                entity["text"] = sentence[entity["start"]: entity["end"]]

                if self.synonym_dict.get(entity["text"], None):
                    entity["original_text"] = entity["text"]
                    entity["text"] = self.synonym_dict[entity["text"]]

            
        

        return predicted_outputs

    def save_pretrained(self, directory: str):
        """
        Save model and tokenizer to directory

        :param directory: path to save folder
        :return: None
        """
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

        


    def train_model(self, dataframe=None, synonym_dict=None,  save_folder: str = "latest_model"):
        """
        Create trainer, train and save best model to save_folder
        :param save_folder: path to save folder
        :return: None
        """
       
        self.synonym_dict.update(synonym_dict)
       
        dataset = DIETClassifierDataset(dataframe=dataframe, tokenizer=self.tokenizer, entities=self.entities[1:], intents=self.intents)
        
        trainer = DIETTrainer(model=self.model, dataset=dataset,
                              train_range=0.95,
                              num_train_epochs=100,
                              per_device_train_batch_size=4,
                              per_device_eval_batch_size=4,
                              warmup_steps=500,
                              weight_decay=0.01,
                              logging_dir="logs",
                              early_stopping_patience=20,
                              early_stopping_threshold=1e-5,
                              output_dir="results")



        trainer.train()

        self.save_pretrained(directory=self.config.get("model"))


if __name__ == "__main__":
    config_file = "/home/lorenzo/Sviluppo/tiledesk/tiledesk-ai/domain/diet/nlu_diet.json"

    try:
        f = open(config_file, "r")
    except Exception as ex:
        raise RuntimeError(f"Cannot read file {config_file} with error:\t{ex}")

    dataforconfig = json.load(f)
    
    nlu = dataforconfig["nlu"]
    dataframe, entities_list,intents_list,synonym_dict = make_dataframe(nlu)

    config={"language":dataforconfig["language"],
                    "pipeline":dataforconfig["configuration"]["pipeline"],
                    "parameters":None,
                    "model":dataforconfig["model"]
    }

    print(config)
    print(dataframe.head(5))
    print(entities_list)
    print(intents_list)
    print(synonym_dict)
    wrapper = DIETClassifierWrapper(config=config,entities_list=entities_list, intents_list=intents_list, synonym_dict=synonym_dict )

    #print(wrapper.predict(["What if I work on office hour"]))
    #print("\n")
    wrapper.train_model(dataframe=dataframe,synonym_dict=synonym_dict )

    #print(wrapper.predict(["afternoon shift please"]))
    #print("\n")
    #print(wrapper.predict(["how about office working hours"]))
    #print("\n")
    #print(wrapper.predict(["How to check attendance?"]))
    