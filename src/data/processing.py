from typing import Tuple, List

import numpy as np
from keras_preprocessing.sequence import pad_sequences

from .mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping


class DataProcessor(object):
    def __init__(self,
                 char_mapping: CharToIdMapping,
                 word_segment_mapping: WordSegmentTypeToIdMapping,
                 bmes_mapping: BMESToIdMapping) -> None:
        self.char_mapping = char_mapping
        self.word_segment_mapping = word_segment_mapping
        self.bmes_mapping = bmes_mapping

    def parse_one(self, sample: str) -> Tuple[np.array, np.array]:
        """
        :param sample: string of the format `упасти	у:PREF/пас:ROOT/ти:SUFF`
        :return: valid tuple X, Y that can be passed to the network input
        """

        def segment_to_label(seg: str, seg_type: str) -> int:
            return len(self.word_segment_mapping) * self.bmes_mapping[seg] + self.word_segment_mapping[seg_type]

        [x, y] = sample.split('\t')
        x = [self.char_mapping[c] for c in x]

        label = []
        y = y.split('/')
        for word_segment in y:
            item = word_segment.split(':')
            assert len(item) == 1 or len(item) == 2
            if len(item) == 1:
                item.append(self.word_segment_mapping.UNK)
            [word_segment, segment_type] = item

            '''Map the word segment to BMES (Begin, Mid, End, Single) encoding '''
            if len(word_segment) == 1:  # Single
                label.append(segment_to_label('S', segment_type))
            else:  # Begin Mid Mid Mid Mid End
                label.append(segment_to_label('B', segment_type))
                label += [segment_to_label('M', segment_type) for _ in word_segment[1: -1]]
                label.append(segment_to_label('E', segment_type))

        assert len(x) == len(label)
        return np.array(x, dtype=np.int32), np.array(label, dtype=np.int32)

    def parse(self, data: List[str]) -> Tuple[np.array, np.array]:
        inputs, labels = [], []
        for sample in data:
            x, y = self.parse_one(sample=sample)
            inputs.append(x)
            labels.append(y)

        inputs = pad_sequences(inputs, truncating='post')
        labels = pad_sequences(labels, truncating='post')
        assert inputs.shape == labels.shape
        return inputs, labels
