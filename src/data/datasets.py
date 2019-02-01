import io
import random
from typing import List, Optional


class Dataset(object):
    def __init__(self,
                 file_path: Optional[str]=None,
                 samples: Optional[List[str]]=None) -> None:

        if file_path is None and samples is None:
            raise ValueError('Either file_path or samples need to be provided')

        if file_path is not None and samples is not None:
            raise ValueError('Cannot combine both file and samples')

        self.data = self.load_data(file_path) if file_path else samples

    @staticmethod
    def load_data(file_path: str) -> List[str]:
        with io.open(file_path, 'r', encoding='utf-8') as f:
            return f.read().splitlines()

    def __len__(self) -> int:
        return len(self.data)

    def shuffle(self) -> None:
        random.shuffle(self.data)

    def __getitem__(self, i: int) -> str:
        return self.data[i]


class BucketDataset(Dataset):
    def __init__(self,
                 file_path: Optional[str] = None,
                 samples: Optional[List[str]] = None) -> None:
        super().__init__(file_path=file_path, samples=samples)

        self.buckets = {}
        for sample in self.data:
            length = len(sample.split('\t')[0])
            if length not in self.buckets:
                self.buckets[length] = []
            self.buckets[length].append(sample)
        self.reorder_buckets()

    def shuffle(self) -> None:
        for key in self.buckets:
            random.shuffle(self.buckets[key])
        self.reorder_buckets()

    def reorder_buckets(self):
        """ Change the order of data to be consistent with buckets """
        data = []
        for bucket_length in sorted(self.buckets.keys()):
            data += self.buckets[bucket_length]
        self.data = data
