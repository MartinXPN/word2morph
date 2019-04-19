from typing import Any, List, Set, Union, Optional


class Mapping(object):
    def __init__(self):
        self.UNK = '<<UNK>>'

    def __getitem__(self, item: Any):
        raise NotImplementedError('You need to implement the mapping operator...')


class KeyToIdMapping(Mapping):
    def __init__(self,
                 keys: Union[List[Union[str, int]], Set[Union[str, int]]],
                 include_unknown: bool = True):

        super(KeyToIdMapping, self).__init__()
        self.keys = [self.UNK] if include_unknown else []
        self.keys += list(keys)
        self.key_to_id = {key: i for i, key in enumerate(self.keys)}

    def get(self, key_id):
        return self.keys[key_id]

    def __getitem__(self, item: Union[str, int]) -> int:
        if item in self.key_to_id:      return self.key_to_id[item]
        if self.UNK in self.key_to_id:  return self.key_to_id[self.UNK]
        raise KeyError('`{}` is not present in the mapping'.format(item))

    def __len__(self) -> int:
        return len(self.keys)

    def __str__(self) -> str:
        return str(self.key_to_id)


class CharToIdMapping(KeyToIdMapping):
    def __init__(self,
                 text: Optional[str] = None,
                 words: Optional[Union[List[str], Set[str]]] = None,
                 chars: Optional[List[str]] = None,
                 include_unknown: bool = True):

        if text is None and words is None and chars is None:
            raise ValueError('Need to provide one of {text, words, chars}')

        chars = dict() if chars is None else dict.fromkeys(chars)
        if text is not None:
            chars.update(dict.fromkeys(text))

        if words is not None:
            for word in dict.fromkeys(words).keys():
                chars.update(dict.fromkeys(word))
        super(CharToIdMapping, self).__init__(list(chars), include_unknown)


class WordSegmentTypeToIdMapping(KeyToIdMapping):
    def __init__(self,
                 segments: Union[List[str], Set[str]],
                 include_unknown: bool = True):
        super(WordSegmentTypeToIdMapping, self).__init__(list(dict.fromkeys(segments)), include_unknown)


class BMESToIdMapping(KeyToIdMapping):
    def __init__(self, begin='B', mid='M', end='E', single='S'):
        self.BEGIN, self.MID, self.END, self.SINGLE = begin, mid, end, single
        super(BMESToIdMapping, self).__init__([self.BEGIN, self.MID, self.END, self.SINGLE], False)


class LabelToIdMapping(KeyToIdMapping):
    def __init__(self,
                 labels: Union[List[int], Set[int]],
                 include_unknown: bool = False):
        super(LabelToIdMapping, self).__init__(list(dict.fromkeys(labels)), include_unknown)
