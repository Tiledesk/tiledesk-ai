from __future__ import annotations
from abc import ABC, abstractmethod

from tileai.core.tiletrainer_ff import TileTrainertorchFF
from tileai.core.tiletrainer_wbag import TileTrainertorchWBag
from tileai.core.tiletrainer_average import TileTrainertorchAverage
from tileai.core.tiletrainer_bag import TileTrainertorchBag
from tileai.core.tiletrainer_bert import TileTrainertorchBert
from tileai.core.tiletrainer_lstm import TileTrainertorchLSTM
from tileai.core.tiletrainer_gpt2 import TileTrainertorchGPT2


class TileTrainerFactory:
    def create_tiletrainer(self, name, language, pipeline, parameters, model):

        if name =='embeddingwbag':
            return TileTrainertorchWBag(language, pipeline, parameters, model)
        elif name == 'embeddigaverage':
            return TileTrainertorchAverage(language, pipeline, parameters, model)
        elif name == 'textclassifier':
            return TileTrainertorchBag(language, pipeline, parameters, model)
        elif name == 'bertclassifier':
            return TileTrainertorchBert(language, pipeline, parameters, model)
        elif name == 'gpt2classifier':
            return TileTrainertorchGPT2(language, pipeline, parameters, model)
        elif name == 'lstm':
            return TileTrainertorchLSTM(language, pipeline, parameters, model)
        elif name == 'diet':
            return TileTrainertorchDIET(language, pipeline, parameters, model)
        else:
            return TileTrainertorchFF(language, pipeline, parameters, model)



"""
 "configuration":{
      "language":"it",
      "pipeline":["bertclassifier","dbmdz/bert-base-italian-uncased"]
    },
"""
   