from typing import Any, Text, Dict, Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tileai.model_training import TrainingResult

def train(nlu:"Text",
          out:"Text")-> "TrainingResult":
    
    from tileai.model_training import train

    return train(nlu,out)

def query(model, query_text):
    from tileai.model_training import query
    return query(model, query_text)

