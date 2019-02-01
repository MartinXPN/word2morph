from src.data.datasets import Dataset
from .processing import DataProcessor


class DataGenerator(object):
    def __init__(self,
                 dataset: Dataset,
                 processor: DataProcessor,
                 batch_size: int,
                 shuffle: bool=True) -> None:
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_start = len(dataset)

    def __next__(self):
        """ Generates a new batch of data """

        ''' Start a new epoch '''
        if self.batch_start >= len(self.dataset):
            self.batch_start = 0
            if self.shuffle:
                self.dataset.shuffle()

        ''' Generate a new batch '''
        batch = [self.dataset[i % len(self.dataset)]
                 for i in range(self.batch_start, self.batch_start + self.batch_size)]
        self.batch_start += self.batch_size
        return self.processor.parse(data=batch)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
