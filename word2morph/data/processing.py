from typing import Tuple, List, Optional

import numpy as np
from keras_preprocessing.sequence import pad_sequences

from word2morph.entities.sample import Sample, Segment
from word2morph.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping, LabelToIdMapping


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
                 label_mapping: Optional[LabelToIdMapping] = None):
        self.char_mapping = char_mapping
        self.word_segment_mapping = word_segment_mapping
        self.bmes_mapping = bmes_mapping
        self.label_mapping = label_mapping

    def label_to_id(self, seg: BMESToIdMapping, seg_type: str) -> int:
        if self.label_mapping:
            return self.label_mapping[(seg, seg_type)]

        return len(self.word_segment_mapping) * self.bmes_mapping[seg] + self.word_segment_mapping[seg_type]

    def segments_to_label(self, segments: Tuple[Segment, ...]) -> List[Tuple[BMESToIdMapping, str]]:
        res = []
        for word_segment in segments:
            word_segment, segment_type = word_segment.segment, word_segment.type
            segment_type = segment_type or self.word_segment_mapping.UNK

            '''Map the word segment to BMES (Begin, Mid, End, Single) encoding '''
            if len(word_segment) == 1:  # Single
                res.append((self.bmes_mapping.SINGLE, segment_type))
            else:  # Begin Mid Mid Mid Mid End
                res.append((self.bmes_mapping.BEGIN, segment_type))
                res += [(self.bmes_mapping.MID, segment_type) for _ in word_segment[1: -1]]
                res.append((self.bmes_mapping.END, segment_type))
        return res

    def parse_one(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param sample: string of the format `упасти	у:PREF/пас:ROOT/ти:SUFF`
        :return: valid tuple X, Y that can be passed to the network input
        """
        x = [self.char_mapping[c] for c in sample.word]
        x = np.array(x, dtype=np.int32)
        y = self.segments_to_label(segments=sample.segments)
        assert len(x) == len(y) or len(y) == 0

        y = [self.label_to_id(seg, seg_type) for seg, seg_type in y]
        y = np.array(y, dtype=np.int32)
        return x, y

    def parse(self, data: List[Sample], convert_one_hot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        inputs, labels = [], []
        for sample in data:
            x, y = self.parse_one(sample=sample)
            inputs.append(x)
            labels.append(y)

        inputs = pad_sequences(inputs, truncating='post')
        labels = pad_sequences(labels, truncating='post')
        assert inputs.shape == labels.shape or labels.shape[1] == 0
        if convert_one_hot:
            labels = np.eye(self.nb_classes())[labels]
        return inputs, labels

    def nb_classes(self) -> int:
        if self.label_mapping is None:
            return len(self.word_segment_mapping) * len(self.bmes_mapping)
        return len(self.label_mapping)

    def to_sample(self, word: str, prediction: np.ndarray) -> Sample:
        """
        The current implementation is a simple greedy algorithm -> take max from all possible values for each position
        :param word: input word (needed so that this method could produce a valid Sample)
        :param prediction: np.array with shape (nb_chars, nb_classes_per_char) -> (9, 25): the output of softmax
        :return: corresponding valid Sample from the prediction
        """
        def is_valid(seg: BMESToIdMapping, seg_type: str):
            try:
                self.label_to_id(seg=seg, seg_type=seg_type)
                return True
            except KeyError:
                return False

        assert len(prediction.shape) == 2
        assert prediction.shape[0] >= len(word) and prediction.shape[1] == self.nb_classes()
        current_seg: Optional[BMESToIdMapping] = None
        current_seg_type: Optional[str] = None
        current_seg_start: int = 0
        segments: List[Segment] = []

        for i, c in enumerate(word):
            ''' Check for Single or Begin for all segment types '''
            if current_seg is None or current_seg in {self.bmes_mapping.END, self.bmes_mapping.SINGLE}:
                single = [-1 if not is_valid(self.bmes_mapping.SINGLE, seg_type) else
                          prediction[i][self.label_to_id(seg=self.bmes_mapping.SINGLE, seg_type=seg_type)]
                          for seg_type in self.word_segment_mapping.keys]
                begin = [-1 if not is_valid(self.bmes_mapping.BEGIN, seg_type) else
                         prediction[i][self.label_to_id(seg=self.bmes_mapping.BEGIN, seg_type=seg_type)]
                         for seg_type in self.word_segment_mapping.keys]

                single_best = np.argmax(single)
                begin_best = np.argmax(begin)
                if max(single) > max(begin):
                    current_seg_type = self.word_segment_mapping.keys[single_best]
                    segments.append(Segment(word[current_seg_start: i+1], segment_type=current_seg_type))
                    current_seg_start = i + 1
                    current_seg = self.bmes_mapping.SINGLE
                else:
                    current_seg_type = self.word_segment_mapping.keys[begin_best]
                    current_seg = self.bmes_mapping.BEGIN

                ''' Check for Mid or End for the current segment type '''
            elif current_seg in {self.bmes_mapping.BEGIN, self.bmes_mapping.MID}:
                if is_valid(self.bmes_mapping.END, current_seg_type) and \
                        prediction[i][self.label_to_id(seg=self.bmes_mapping.END, seg_type=current_seg_type)] > \
                        prediction[i][self.label_to_id(seg=self.bmes_mapping.MID, seg_type=current_seg_type)]:
                    segments.append(Segment(word[current_seg_start: i+1], segment_type=current_seg_type))
                    current_seg_start = i + 1
                    current_seg = self.bmes_mapping.END
                else:
                    current_seg = self.bmes_mapping.MID

        if current_seg_start < len(word):
            segments.append(Segment(word[current_seg_start:], segment_type=current_seg_type))
        return Sample(word=word, segments=tuple(segments))
