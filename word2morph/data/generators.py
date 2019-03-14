from typing import Tuple, Generator

import numpy as np

from word2morph.entities.dataset import Dataset
from .processing import DataProcessor


class DataGenerator(object):
    def __init__(self,
                 dataset: Dataset,
                 processor: DataProcessor,
                 batch_size: int,
                 shuffle: bool=True):
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size

        if shuffle:
            self.dataset.shuffle()

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for start in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[start: start + self.batch_size]
            yield self.processor.parse(data=batch)

    def __len__(self):
        if len(self.dataset) % self.batch_size == 0:
            return len(self.dataset) // self.batch_size
        return len(self.dataset) // self.batch_size + 1
