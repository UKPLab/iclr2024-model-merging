import itertools
from abc import ABC, abstractmethod


class Preprocessor(ABC):

    def __init__(self, config, data_args, model_args, tokenizer):
        self.config = config
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess(self, features):
        pass
