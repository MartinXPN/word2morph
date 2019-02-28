from typing import Optional, List

from src.entities.sample import Segment, Sample


class DataLoader(object):
    """
    Loads string/file data into Sample or list of Samples
    """
    def __init__(self,
                 file_path: Optional[str] = None,
                 samples: Optional[List[str]] = None):
        if file_path is None and samples is None:
            raise ValueError('Either file_path or samples need to be provided')

        if file_path is not None and samples is not None:
            raise ValueError('Cannot combine both file and samples')

        self.data: List[str] = self.load_file(file_path) if file_path else samples

    def load(self) -> List[Sample]:
        return [self.parse_one(item=item) for item in self.data]

    @staticmethod
    def load_file(file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().splitlines()

    @staticmethod
    def parse_one(item: str) -> Sample:
        # `упасти	у:PREF/пас:ROOT/ти:SUFF`
        [word, label] = item.split('\t')
        segments = []
        for word_segment in label.split('/'):
            part = word_segment.split(':')
            assert 1 <= len(part) <= 2
            if len(part) == 1:
                segments.append(Segment(segment=part[0]))
            else:
                segments.append(Segment(segment=part[0], segment_type=part[1]))

        return Sample(word=word, segments=tuple(segments))
