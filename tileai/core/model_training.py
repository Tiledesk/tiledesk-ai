import json
import logging
from typing import Text, NamedTuple, Optional, List, Union, Dict, Any
import uuid
import numpy as np
import os

import asyncio



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
    
    from tileai.core.preprocessing.data_reader import make_dataframe
    df, entities_list, intents_list, synonym_dict = make_dataframe(nlu_json)


    

    #train_texts=[]
    #train_labels=[]
   

    

    #for obj in nlu_json:    
    #    for example in obj["examples"]:
    #        train_texts.append(example) 
    #        train_labels.append(obj["intent"])
            
    #logger.info(train_texts)
    #logger.info(train_labels)

    #print(train_texts)
    #print(train_labels)    
       
    if not os.path.exists(out):
        os.makedirs(out)

    #if pipeline[0] == "textclassifier":
    #    tiletrainertorch = TileTrainertorchBag("it", pipeline, "", None)
    #else:
    #    tiletrainertorch = TileTrainertorchFF("it",pipeline, "",None)

    tiletrainerfactory = TileTrainerFactory()
    tiletrainertorch = tiletrainerfactory.create_tiletrainer(pipeline[0],language,pipeline, "",model=out )
 
    #report = tiletrainertorch.train(train_texts, train_labels)
    report = tiletrainertorch.train(df, entities_list, intents_list, synonym_dict)
    
        
       
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


    label, result_dict = tiletrainertorch.query(config,  query_text)
    return label,result_dict
    
    
async def http_query(redis_conn, model, query_text):
    #model models/models/diet-test-status il path al modello e relative configurazioni
    import dill
   
    async with redis_conn as r:
        
        dill_redis_model = await r.get(model)
        redmc = await r.get(model+"/bin")
    
    if dill_redis_model is None:
        #scrivo su redis
        print("load from file system")
        json_filename = model+"/"+const.MODEL_CONFIG
        jsonfile_config = open (json_filename, "r", encoding='utf-8')
        config = json.loads(jsonfile_config.read())
        jsonfile_config.close()
        
    
        pipeline= config["pipeline"]
        tiletrainerfactory = TileTrainerFactory()
        tiletrainertorch = tiletrainerfactory.create_tiletrainer(pipeline[0],"it",pipeline, "",model )
        if pipeline[0]=="bertclassifier":
            print("bert load from file system")
            from tileai.core.preprocessing.textprocessing import load_model_bert
            from tileai.core.http.redis_model import RedisModel
            model_classifier, id2label, tokenizer  = load_model_bert(config, model)
            redis_model = RedisModel(vocabulary=None,
                                     id2label=id2label,
                                     configuration=config, 
                                     tokenizer=tokenizer, 
                                     #model=model_classifier,
                                     tiletrainertorch= tiletrainertorch
                                     )
            
            redmc = dill.dumps(model_classifier)
            dill_redis_model = dill.dumps(redis_model)

            async with redis_conn as r:
                await r.set(model, dill_redis_model)
                await r.set(model+"/bin", redmc) 
        
        elif pipeline[0]=="diet":
            print("DIET load formfile system")
            
            from tileai.core.classifier.diet_wrapper import DIETClassifierWrapper
            from tileai.core.http.redis_model import RedisModel
            model_classifier = DIETClassifierWrapper(
               config={"language":config.get("language", None),
                        "pipeline":config.get("pipeline", None),
                        "parameters":config.get("parameters", None),
                        "model":model
            },entities_list=config.get("entities", None), intents_list=config.get("intents", None), synonym_dict=config.get("synonim", None))
           
            redis_model = RedisModel(vocabulary=None,
                                     id2label=None,
                                     configuration=config, 
                                     tokenizer=None, 
                                     #model=model_classifier,
                                     tiletrainertorch= tiletrainertorch
                                     )
            
            redmc = dill.dumps(model_classifier)
            dill_redis_model = dill.dumps(redis_model)

            async with redis_conn as r:
                await r.set(model, dill_redis_model)
                await r.set(model+"/bin", redmc) 

        elif pipeline[0]=="semantic":
            print("Semantic Similarity")
            from tileai.core.http.redis_model import RedisModel
            redis_model = RedisModel(vocabulary=None,
                                     id2label=None,
                                     configuration=config, 
                                     tokenizer=None, 
                                     #model=model_classifier,
                                     tiletrainertorch= None
                                     )
            model_classifier = config["model"]

        else:    
            from tileai.core.preprocessing.textprocessing import load_model
            from tileai.core.tokenizer.standard_tokenizer import StandarTokenizer
            from tileai.core.http.redis_model import RedisModel
            vocabll,model_classifier, id2label  = load_model(config, model)
            #dill_class_model = dill.dumps(model_classifier)
            tokenizer = StandarTokenizer()#get_tokenizer("basic_english") 
            #vocabulary, model, configuration, tokenizer, id2label
            redis_model = RedisModel(vocabulary=vocabll,
                                     id2label=id2label,
                                     configuration=config, 
                                     tokenizer=tokenizer, 
                                     #model=model_classifier,
                                     tiletrainertorch= tiletrainertorch
                                     )

            redmc = dill.dumps(model_classifier)
            dill_redis_model = dill.dumps(redis_model)
       
            async with redis_conn as r:
                await r.set(model, dill_redis_model)
                await r.set(model+"/bin", redmc) 
        mod=model_classifier
        
    else:
        #leggo da redis
        redis_model= dill.loads(dill_redis_model) 
        conf = redis_model.configuration
        pipeline= conf["pipeline"]
        tiletrainertorch = redis_model.tiletrainertorch
        #model_classifier=dill.loads(redis_model.model)
        mod = dill.loads(redmc)
        
        
        
        #model_classifier.eval()
        #tiletrainerfactory = TileTrainerFactory()
        #tiletrainertorch = tiletrainerfactory.create_tiletrainer(pipeline[0],"it",pipeline, "",model )

            
    
    
    label, result_dict = tiletrainertorch.query_http(vocabll=redis_model.vocabulary,
                                                     model_classifier=mod,
                                                     id2label=redis_model.id2label,
                                                     tokenizer=redis_model.tokenizer, 
                                                     query_text=query_text)
    
    return label,result_dict    
    
    
    
async def del_old_model(redis_conn, model):
    async with redis_conn as r:
        await r.delete(model+"/bin")
        await r.delete(model)


