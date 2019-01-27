import random
from typing import List

from .processing import DataProcessor


class DataGenerator(object):
    def __init__(self,
                 samples: List[str],
                 processor: DataProcessor,
                 batch_size: int,
                 shuffle: bool=True) -> None:
        self.samples = samples
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_start = len(samples)
        self.indices = list(range(len(samples)))

    def __next__(self):
        """ Generates a new batch of data """

        ''' Start a new epoch '''
        if self.batch_start >= len(self.samples):
            self.batch_start = 0
            if self.shuffle:
                random.shuffle(self.indices)

        ''' Generate a new batch '''
        batch = [self.samples[i] for i in self.indices[self.batch_start: self.batch_start + self.batch_size]]
        self.batch_start += self.batch_size
        return self.processor.parse(data=batch)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
