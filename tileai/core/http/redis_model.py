import logging

logger = logging.getLogger(__name__)

class RedisModel:
    def __init__(self, vocabulary, id2label,configuration,tokenizer,tiletrainertorch ):# model, , tokenizer, id2label ):
        self.vocabulary=vocabulary
        self.id2label = id2label
        self.configuration = configuration
        self.tokenizer = tokenizer
        #self.model = model,
        self.tiletrainertorch=tiletrainertorch

       
        
    
    def load_model(self):
        pass

