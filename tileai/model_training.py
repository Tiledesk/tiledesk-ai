import json
import logging
from typing import Text, NamedTuple, Optional, List, Union, Dict, Any
import uuid
import numpy as np
import os

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
   
     
    nlu_json =  nlu["nlu"]
   
    if "configuration" not in nlu or "algo" not in nlu["configuration"]:
        algo="feedforward"
    else:
        algo = nlu["configuration"]["algo"]
    # Riga aggiunta per non avere errori sulla varibile
    
    train_texts=[]
    train_labels=[]
    intents=[]
    for obj in nlu_json:    
        for example in obj["examples"]:
            train_texts.append(example) 
            train_labels.append(obj["intent"])
            
    logger.info(train_texts)
    logger.info(train_labels)
        
       
    if not os.path.exists(out):
        os.makedirs(out)

    tiletrainertorch = TileTrainertorchFF("it",algo, "dd",None)
    state_dict, configdata, vocab, report = tiletrainertorch.train(train_texts, train_labels)
    torch.save (state_dict, out+"/model.bin")

    config_json = out+"/config.json"
    vocab_file = out+"/vocab.vby"
    print(config_json)
    
   

    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(configdata, f, ensure_ascii=False, indent=4)
    
    f.close()
    print(vocab)
    with open(vocab_file, 'w', encoding='utf-8') as f_v:
        for vb in vocab:
            f_v.write(vb)
            f_v.write("\n")
           
    f_v.close()
       
    return TrainingResult(str(out), 0)
    
def query(model, query_text):
    
    json_filename = model+"/config.json"
    jsonfile_config = open (json_filename, "r", encoding='utf-8')
    config = json.loads(jsonfile_config.read())
    jsonfile_config.close()

    vocabulary = []
    vocab_file = model+"/vocab.vby"
    vocabulary = open (vocab_file, "r",  encoding='utf-8').read().splitlines()
    
    modelname =   model+"/model.bin"
    
    

    tiletrainertorch = TileTrainertorchFF("it","dbmdz/bert-base-italian-xxl-cased", "dd",None)

    label, model, vocab, result_dict = tiletrainertorch.query(modelname, config, vocabulary, query_text)
    return label,result_dict
    
    
    
    
    
