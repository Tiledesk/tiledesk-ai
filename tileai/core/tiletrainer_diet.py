






class TileTrainertorchDIET(TileTrainer):




    def __init__(self, language, pipeline, parameters, model):
        self.language=language
        self.pipeline=pipeline
        self.parameters=parameters
        self.model = model
    
    def train(self, train_texts,train_labels):
        pass


    def query(self, configuration, query_text):
        pass

    def query_http(self, vocabll,model_classifier, id2label,tokenizer, query_text):
        pass