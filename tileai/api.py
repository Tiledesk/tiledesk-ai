from typing import Any, Text, Dict, Union, List, Optional
import json
import tileai.shared.utils


    


def train(nlu:"Text",
          out:"Text")-> "TrainingResult":
    
    from tileai.core.model_training import train

    with open(nlu) as jsonFile:
       jsonObject = json.load(jsonFile)
       jsonFile.close()

    return train(jsonObject,out)

def query(model, query_text):
    from tileai.core.model_training import query
    return query(model, query_text)

