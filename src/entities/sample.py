from typing import Tuple, Optional


class Segment(object):
    # дум:ROOT
    def __init__(self, segment: str, segment_type: str = None):
        self.segment = segment      # дум
        self.type = segment_type    # ROOT

    def __str__(self):
        return '({}:{})'.format(self.segment, self.type)


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
        return '{} -> ({})'.format(self.word, ', '.join([str(segment) for segment in self.segments]))
