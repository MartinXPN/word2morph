from typing import Tuple, Union, List

import numpy as np
from keras.utils import Sequence

from word2morph.entities.dataset import Dataset
from word2morph.entities.sample import Sample
from .processing import DataProcessor


class DataGenerator(Sequence):
    def __init__(self,
                 dataset: Dataset,
                 processor: DataProcessor,
                 batch_size: int,
                 with_samples: bool = False,
                 shuffle: bool = True,
                 drop_remainder: bool = False):
        self.dataset = dataset
        self.processor = processor
        self.with_samples = with_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder

        if shuffle:
            self.shuffle_data()

    def __len__(self) -> int:
        """ Number of batches to be generated from the dataset """
        if len(self.dataset) % self.batch_size == 0:
            return len(self.dataset) // self.batch_size
        return len(self.dataset) // self.batch_size + int(not self.drop_remainder)

    def __getitem__(self, index: int) -> Union[Tuple[np.ndarray, np.ndarray],
                                               Tuple[np.ndarray, np.ndarray, List[Sample]]]:
        """ Gets batch at position `index` """
        batch = self.dataset[index * self.batch_size: (index + 1) * self.batch_size]
        res = self.processor.parse(data=batch)

        if self.with_samples:
            return res + (batch,)
        return res

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        assert self.shuffle
        self.dataset.shuffle()
