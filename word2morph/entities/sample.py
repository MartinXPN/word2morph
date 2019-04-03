from typing import Tuple, Optional


class Segment(object):
    # дум:ROOT
    def __init__(self, segment: str, segment_type: str = None):
        self.segment = segment      # дум
        self.type = segment_type    # ROOT

    def __str__(self):
        if self.type is None:
            return f'{self.segment}'
        return f'{self.segment}:{self.type}'

    def __eq__(self, other: 'Segment') -> bool:
        return self.segment == other.segment and self.type == other.type


class Sample(object):
    # word=одуматься	segments=о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX
    # word=одуматься    segments=((о, PREF), (дум, ROOT), (а, SUFF), (ть, SUFF), (ся, POSTFIX))
    def __init__(self, word: str, segments: Tuple[Segment, ...]):
        self.word = word
        self.segments = segments

    @property
    def segment_types(self) -> Tuple[Optional[str], ...]:
        return tuple([segment.type for segment in self.segments])

    @property
    def segment_parts(self) -> Tuple[str, ...]:
        return tuple([segment.segment for segment in self.segments])

    def __str__(self):
        segments_str = '/'.join([str(segment) for segment in self.segments])
        return f'{self.word}\t{segments_str}'

    def __eq__(self, other: 'Sample') -> bool:
        return self.word == other.word and self.segments == other.segments
