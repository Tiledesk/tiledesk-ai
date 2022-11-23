import abc
import re 
import logging

from typing import Text, List #, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Tokenizer(abc.ABC):
    def __init__(self, )->None:
        pass
    
    @abc.abstractmethod
    def tokenize(self, input_string: Text) -> List[Text]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        ...
