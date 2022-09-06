import json
import logging
from typing import Text, NamedTuple, Optional, List, Union, Dict, Any
import uuid

from tileai.TileTrainertorchFF import TileTrainertorchFF
import torch

logger = logging.getLogger(__name__)


class TrainingResult(NamedTuple):
    """Holds information about the results of training."""

    model: Optional[Text] = None
    code: int = 0
    #dry_run_results: Optional[Dict[Text, Union[uuid.uuid4().hex, Any]]] = None

def train(nlu:"Text",
          out:"Text")-> TrainingResult:
    """Trains a Tileai model with NLU data.

    Args:
        nlu: Path to the nlu file.
        out: Path to the output model directory.
        

    Returns:
        An instance of `TrainingResult`.
    """
    

    #Leggo il file nlu
    with open(nlu) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
     
    nlu_json =  jsonObject["nlu"]
    train_texts=[]
    train_labels=[]
    intents=[]
    for obj in nlu_json:    
        for example in obj["examples"]:
            train_texts.append(example) 
            train_labels.append(obj["intent"])
            
    logger.info(train_texts)
    logger.info(train_labels)
        
       
       
    tiletrainertorch = TileTrainertorchFF("it","dbmdz/bert-base-italian-xxl-cased", "dd",None)
    state_dict = tiletrainertorch.train(train_texts, train_labels)
    torch.save (state_dict, out)
    
       
    return tiletrainertorch
    
   
