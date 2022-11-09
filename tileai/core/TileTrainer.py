import json
import numpy as np
import re
from tqdm import tqdm
import gc
from collections import OrderedDict
import time
import importlib
import logging

class TileTrainer:

    """
    
    """    

    def __init__(self, language, pipeline, parameters, model):
        self.language=language
        self.pipeline=pipeline
        self.parameters=parameters
        self.model = model