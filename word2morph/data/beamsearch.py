from math import log, inf
from typing import Tuple, List, Optional

import numpy as np

from word2morph.data.processing import DataProcessor
from word2morph.entities.sample import Sample, Segment
from word2morph.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping, LabelToIdMapping


class PostProcessBeamSearch(DataProcessor):

    def __init__(self,
                 char_mapping: CharToIdMapping,
                 word_segment_mapping: WordSegmentTypeToIdMapping,
                 bmes_mapping: BMESToIdMapping,
                 label_mapping: LabelToIdMapping):
        self.char_mapping = char_mapping
        self.word_segment_mapping = word_segment_mapping
        self.bmes_mapping = bmes_mapping
        self.label_mapping = label_mapping

    def penalty(self, sequence: List[int], cur_id: int, seq_score: float, cur_prob: float) -> float:
        """
        Penalty for the sequence with next predicted value
        :param sequence: the ids of the whole sequence
        :param cur_id: the last element which is to be scored against adding it to the sequence
        :param seq_score: current score of the sequence
        :param cur_prob: predicted probability of having cur_id at this position
        :return: penalty for cur_id being added to the sequence
        """
        ok = True
        if sequence:
            # M,        ROOT
            cur_seg, cur_type = self.label_mapping.keys[cur_id]
            prev_seg, prev_type = self.label_mapping.keys[sequence[-1]]

            # Check start of a new sequence
            if prev_type != cur_type:
                if (prev_seg == BMESToIdMapping.SINGLE and cur_seg in {BMESToIdMapping.MID, BMESToIdMapping.END}) or \
                        (prev_seg == BMESToIdMapping.BEGIN) or \
                        (prev_seg == BMESToIdMapping.MID) or \
                        (prev_seg == BMESToIdMapping.END and cur_seg in {BMESToIdMapping.MID, BMESToIdMapping.END}):
                    ok = False
            else:
                if (prev_seg == BMESToIdMapping.SINGLE and cur_seg in {BMESToIdMapping.MID, BMESToIdMapping.END}) or \
                        (prev_seg == BMESToIdMapping.BEGIN and cur_seg in {BMESToIdMapping.SINGLE, BMESToIdMapping.BEGIN}) or \
                        (prev_seg == BMESToIdMapping.MID and cur_seg in {BMESToIdMapping.SINGLE, BMESToIdMapping.BEGIN}) or \
                        (prev_seg == BMESToIdMapping.END and cur_seg in {BMESToIdMapping.MID, BMESToIdMapping.END}):
                    ok = False

        return seq_score * -log(cur_prob) if ok and cur_prob else inf

    def beam_search(self, data: np.ndarray, k: int) -> List[Tuple[List[int], float]]:
        sequences: List[Tuple[List[int], float]] = [(list(), 1.0)]
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = (seq + [j], self.penalty(sequence=seq, cur_id=j, seq_score=score, cur_prob=row[j]))
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences

    def to_sample(self, word: str, prediction: np.ndarray) -> Sample:
        """
        :param word: input word (needed so that this method could produce a valid Sample)
        :param prediction: np.array with shape (nb_chars, nb_classes_per_char) -> (9, 25): the output of softmax
        :return: corresponding valid Sample from the prediction
        """

        assert len(prediction.shape) == 2
        assert prediction.shape[0] >= len(word) and prediction.shape[1] == len(self.label_mapping)

        # make elements of padding equal to 0
        for i in range(len(word), prediction.shape[0]):
            prediction[i] = 0
        sequences = self.beam_search(data=prediction[:len(word)], k=3)

        segments: List[Segment] = []
        current_seg: Optional[BMESToIdMapping] = None
        current_seg_type: Optional[str] = None
        current_seg_start: int = 0
        for i in range(len(word)):
            label_id = sequences[0][0][i]
            if current_seg is None or current_seg_type is None:
                current_seg, current_seg_type = self.label_mapping.keys[label_id]

            current_seg, _ = self.label_mapping.keys[label_id]

            if current_seg in {BMESToIdMapping.SINGLE, BMESToIdMapping.END}:
                segments.append(Segment(word[current_seg_start: i + 1], segment_type=current_seg_type))
                current_seg_start = i + 1
                current_seg, current_seg_type = None, None

            # Only the selected prediction should be 1 others become 0
            prediction[i] = 0
            prediction[i][label_id] = 1

        if current_seg is not None:
            segments.append(Segment(word[current_seg_start:], segment_type=current_seg_type))

        return Sample(word=word, segments=tuple(segments))
