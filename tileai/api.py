from typing import Any, Text, Dict, Union, List, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from tileai.model_training import TrainingResult


def run(port:5100):
    
    from tileai.http.httprun import serve_application

    _endpoints = "0.0.0.0"
    
    kwargs = {}
    serve_application(
        model_path="models/new",
        endpoints=_endpoints,
        port=port,
        **kwargs
    )
    print("RUN", port)


def train(nlu:"Text",
          out:"Text")-> "TrainingResult":
    
    from tileai.model_training import train

    with open(nlu) as jsonFile:
       jsonObject = json.load(jsonFile)
       jsonFile.close()

    return train(jsonObject,out)

def query(model, query_text):
    from tileai.model_training import query
    return query(model, query_text)

