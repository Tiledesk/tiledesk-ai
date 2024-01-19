import json
from typing import List, Dict, Text, Any, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

class CRFClassifierDataset(Dataset):
    def __init__(self,  texts, tags):
        
        self.texts = texts
        self.tags = tags

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index):
        item = {
            'text': self.texts[index],
            'tags' :self.tags[index]
        
        }

        return item #self.texts[index],self.tags[index]
    # def __getitem__(self, index):
    #     text_tensor = torch.tensor(self.texts[index])  # Convert to tensor
    #     tags_tensor = torch.tensor(self.tags[index])
    #     return {
    #         'text': text_tensor,
    #         'tags': tags_tensor
    #     }





def _remove_entities(entities_list: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    
    if isinstance(entities_list, str):
        try:
            entities_list = json.loads(entities_list)
            
        except Exception as ex:
            raise RuntimeError(f"Cannot convert entity {entities_list} by error: {ex}")
    
    entities_list = [entity for entity in entities_list if entity["entity_name"] in org_enti]

    return entities_list
    
def _process_entities_row(row):
        return {
            'text': row['example'],
            'tags': row['entities'],
        }

def make_list_entities_tags(dataframe: pd.DataFrame, entities: List[str]):

    """
    list of entities and tags from dataframe

    :param dataframe: dataframe contains ["example", "intent", "entities"] columns
    :param entities: list of entities class names

    """
       
    
    entities = ["O"] + entities
   
    global org_enti 
    org_enti = entities
    dataframe["entities"] = dataframe["entities"].apply(_remove_entities,entities )

    num_entities = len(entities)
        

    sentences = dict(
        sentence=[],
        entities=[],
            
    )

    
    for _, row in dataframe.iterrows():
        sentences["sentence"].append(row["example"])
        sentences["entities"].append(row["entities"])

               
    

    #print(sentences)
    words = []
    labels = []

    for sentence, entitiesm in zip(sentences["sentence"],sentences["entities"]):
        
        if not entitiesm: 
            wordlist = sentence.split()
            words.extend([wordlist])
            labels.extend([['O'] * len(wordlist)])
        else:
            #label = 'O'  # 'O' means no entity
            for entity in entitiesm:   
                before, _, after = sentence.partition(entity["entity"])
                before_words = before.split()
                after_words = after.split()
                #sentence = sentence[:start] + entity_text + sentence[end:]
                words.extend([before_words + [entity["entity"]] + after_words])
                labels.extend([['O'] * len(before_words) + [entity["entity_name"]] + ['O'] * len(after_words)])
                      
 
               
        
    #print("Words:", words)
    #print("Labels:", labels)
    
    return words, labels


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())

    from tileai.core.preprocessing.data_reader import make_dataframe, read_from_json
    #from transformers import AutoTokenizer
    from tileai.core.tokenizer.standard_tokenizer import StandarTokenizer

    files = ["/home/lorenzo/Sviluppo/tiledesk/tiledesk-ai/domain/diet/nlu_diet.json"]

    data=[]
    for file in files:
        data += read_from_json(file=file)

    #tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    tokenizer = StandarTokenizer()


    df, entities_list, intents_list, synonym_dict = make_dataframe(data)
    words, tags = make_list_entities_tags(dataframe=df, entities=entities_list)

    print(len(words),len(tags))
    print(words[100])
    print(tags[100])
    
