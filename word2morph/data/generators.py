from typing import Tuple, Generator, Union, List

import numpy as np

from word2morph.entities.dataset import Dataset
from word2morph.entities.sample import Sample
from .processing import DataProcessor


class DataGenerator(object):
    def __init__(self,
                 dataset: Dataset,
                 processor: DataProcessor,
                 batch_size: int,
                 with_samples: bool = False,
                 shuffle: bool = True):
        self.dataset = dataset
        self.processor = processor
        self.with_samples = with_samples
        self.batch_size = batch_size

        if shuffle:
            self.dataset.shuffle()

    def __iter__(self) -> Generator[Union[Tuple[np.ndarray, np.ndarray],
                                          Tuple[np.ndarray, np.ndarray, List[Sample]]],
                                    None,
                                    None]:
        for start in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[start: start + self.batch_size]
            res = self.processor.parse(data=batch)
            if self.with_samples:
                res = res + (batch,)
                yield res
            else:
                yield res

    def __len__(self):
        if len(self.dataset) % self.batch_size == 0:
            return len(self.dataset) // self.batch_size
        return len(self.dataset) // self.batch_size + 1
