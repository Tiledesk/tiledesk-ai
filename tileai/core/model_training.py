import json
import logging
from typing import Text, NamedTuple, Optional, List, Union, Dict, Any
import uuid
import numpy as np
import os



import tileai.shared.const as const
from tileai.core.tiletrainer import TileTrainerFactory

logger = logging.getLogger(__name__)


class TrainingResult(NamedTuple):
    """Holds information about the results of training."""

    model: Optional[Text] = None
    code: int = 0
    performanceindex : Optional[Dict] = None
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
   
    if "configuration" not in nlu or "pipeline" not in nlu["configuration"]:
        pipeline="feedforward"
    else:
        pipeline = nlu["configuration"]["pipeline"]
    # Riga aggiunta per non avere errori sulla varibile

    if "configuration" not in nlu or "language" not in nlu["configuration"]:
        language=""
    else:
        language = nlu["configuration"]["language"]
    
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

    #if pipeline[0] == "textclassifier":
    #    tiletrainertorch = TileTrainertorchBag("it", pipeline, "", None)
    #else:
    #    tiletrainertorch = TileTrainertorchFF("it",pipeline, "",None)

    tiletrainerfactory = TileTrainerFactory()
    tiletrainertorch = tiletrainerfactory.create_tiletrainer(pipeline[0],language,pipeline, "",model=out )
 
    report = tiletrainertorch.train(train_texts, train_labels)
    
        
       
    return TrainingResult(str(out), 0, report)
    
def query(model, query_text):
    
    ### TO FIX 
    # Anche per la query è necessario verificare il tipo  di modello e creare l'istanza giusta per la preparazione del dato
    json_filename = model+"/"+const.MODEL_CONFIG
    jsonfile_config = open (json_filename, "r", encoding='utf-8')
    config = json.loads(jsonfile_config.read())
    jsonfile_config.close()

    
    
    
    
    pipeline= config["pipeline"]

    tiletrainerfactory = TileTrainerFactory()
    tiletrainertorch = tiletrainerfactory.create_tiletrainer(pipeline[0],"it",pipeline, "",model )

    #tiletrainertorch = TileTrainertorchFF("it","", "dd",None)

    label, result_dict = tiletrainertorch.query(config,  query_text)
    return label,result_dict
    
    
    
    
    
