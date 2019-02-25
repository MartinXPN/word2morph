from typing import Tuple, Iterable, Generator

import numpy as np

from src.entities.dataset import Dataset
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
        def gen(indices: Iterable):
            batch = [self.dataset[i % len(self.dataset)]
                     for i in indices]
            return self.processor.parse(data=batch)

        start = 0
        current_batch = gen(range(0, self.batch_size))
        while True:
            print('Generating batch with start:', start)
            yield current_batch
            start += self.batch_size
            start %= len(self.dataset)
            current_batch = gen(range(start, start+self.batch_size))
