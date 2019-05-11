from typing import List, Set, Union, Optional, Iterable, Tuple, TypeVar, Generic

T = TypeVar('T')


class Mapping(Generic[T]):
    def __init__(self):
        self.UNK = '<<UNK>>'

    def __getitem__(self, item: T):
        raise NotImplementedError('You need to implement the mapping operator...')


class KeyToIdMapping(Mapping):
    def __init__(self,
                 keys: Iterable[T],
                 include_unknown: bool = True):

        super().__init__()
        self.keys: List[T] = [self.UNK] if include_unknown else []
        self.keys += list(keys)
        self.key_to_id = {key: i for i, key in enumerate(self.keys)}

    def get(self, key_id: int):
        return self.keys[key_id]

    def __getitem__(self, item: T) -> int:
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
                 words: Optional[Iterable[str]] = None,
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
        super().__init__(list(chars), include_unknown)


class WordSegmentTypeToIdMapping(KeyToIdMapping):
    def __init__(self,
                 segments: Union[List[str], Set[str]],
                 include_unknown: bool = True):
        super().__init__(list(dict.fromkeys(segments)), include_unknown)


class BMESToIdMapping(KeyToIdMapping):
    BEGIN = 'B'
    MID = 'M'
    END = 'E'
    SINGLE = 'S'

    def __init__(self):
        super(BMESToIdMapping, self).__init__([self.BEGIN, self.MID, self.END, self.SINGLE], False)


class LabelToIdMapping(KeyToIdMapping):
    def __init__(self,
                 labels: Union[Iterable[Tuple[BMESToIdMapping, str]],
                               Iterable[BMESToIdMapping]],
                 include_unknown: bool = False):
        """
        :param labels: [BMES, segment.type]
        """
        super().__init__(keys=list(dict.fromkeys(labels)), include_unknown=include_unknown)
