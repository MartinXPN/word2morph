import random
from typing import List, Set, overload, Iterator

from .sample import Sample


class Dataset(object):
    def __init__(self, samples: List[Sample]):
        self.data = samples

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.data)

    def shuffle(self):
        random.shuffle(self.data)

    @overload
    def __getitem__(self, i: slice) -> List[Sample]:
        ...

    @overload
    def __getitem__(self, i: int) -> Sample:
        ...

    def __getitem__(self, i) -> Sample:
        return self.data[i]

    def get_chars(self) -> Set[str]:
        chars = set()
        for sample in self.data:
            chars.update(sample.word)
        return chars

    def get_segment_types(self) -> Set[str]:
        segments = set()
        for sample in self.data:
            segments.update(sample.segment_types)

        if None in segments:
            segments.remove(None)
        return segments


class BucketDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        super().__init__(samples=samples)

        self.buckets = {}
        for sample in self.data:
            length = len(sample.word)
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
