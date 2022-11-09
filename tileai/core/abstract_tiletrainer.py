from __future__ import annotations
from abc import ABC, abstractmethod, ABCMeta
class TileTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def query(self):
        pass