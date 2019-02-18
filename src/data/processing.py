from typing import Tuple, List

import numpy as np
from keras_preprocessing.sequence import pad_sequences

from src.entities.sample import Sample
from .mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping, LabelToIdMapping


class DataProcessor(object):
    """
    Maps data in both directions (1. from sample to network 2. from network to sample)
    1. Generates network input/label from a Sample
    2. Generates a valid Sample from network input and prediction
    """
    def __init__(self,
                 char_mapping: CharToIdMapping,
                 word_segment_mapping: WordSegmentTypeToIdMapping,
                 bmes_mapping: BMESToIdMapping,
                 label_mapping: LabelToIdMapping = None):
        self.char_mapping = char_mapping
        self.word_segment_mapping = word_segment_mapping
        self.bmes_mapping = bmes_mapping
        self.label_mapping = label_mapping

    def parse_one(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param sample: string of the format `упасти	у:PREF/пас:ROOT/ти:SUFF`
        :return: valid tuple X, Y that can be passed to the network input
        """

        def segment_to_label(seg: str, seg_type: str) -> int:
            res = len(self.word_segment_mapping) * self.bmes_mapping[seg] + self.word_segment_mapping[seg_type]
            if self.label_mapping:
                res = self.label_mapping[res]
            return res

        x = [self.char_mapping[c] for c in sample.word]
        y = []

        for word_segment in sample.segments:
            word_segment, segment_type = word_segment.segment, word_segment.type
            segment_type = segment_type or self.word_segment_mapping.UNK

            '''Map the word segment to BMES (Begin, Mid, End, Single) encoding '''
            if len(word_segment) == 1:  # Single
                y.append(segment_to_label('S', segment_type))
            else:  # Begin Mid Mid Mid Mid End
                y.append(segment_to_label('B', segment_type))
                y += [segment_to_label('M', segment_type) for _ in word_segment[1: -1]]
                y.append(segment_to_label('E', segment_type))

        assert len(x) == len(y)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def parse(self, data: List[Sample], convert_one_hot: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        inputs, labels = [], []
        for sample in data:
            x, y = self.parse_one(sample=sample)
            inputs.append(x)
            labels.append(y)

        inputs = pad_sequences(inputs, truncating='post')
        labels = pad_sequences(labels, truncating='post')
        assert inputs.shape == labels.shape
        if convert_one_hot:
            labels = np.eye(self.nb_classes())[labels]
        return inputs, labels

    def nb_classes(self) -> int:
        if self.label_mapping is None:
            return len(self.word_segment_mapping) * len(self.bmes_mapping)
        return len(self.label_mapping)

    def to_sample(self, word: str, prediction: np.ndarray) -> Sample:
        """
        :param word: input word (needed so that this method could produce a valid Sample)
        :param prediction: np.array with shape (nb_chars, nb_classes_per_char) -> (9, 25) or (classes,) -> (9,)
                            for the first option the prediction should be the output of softmax
                            for the second option the prediction should contain class_ids for each character
        :return: corresponding valid Sample from the prediction
        """
        assert 1 <= len(prediction.shape) <= 2
        classes = np.argmax(prediction, axis=1) if len(prediction.shape) == 2 else prediction
        return Sample(word='asdf', segments=())
