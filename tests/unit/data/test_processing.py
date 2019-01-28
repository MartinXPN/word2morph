from unittest import TestCase

import numpy as np

from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping
from src.data.processing import DataProcessor


class TestPreprocessing(TestCase):

    def test_processing(self):
        char_mapping = CharToIdMapping(text='одуматься', include_unknown=True)
        word_mapping = WordSegmentTypeToIdMapping(segments=['PREF', 'ROOT', 'SUFF'])

        print('Char mapping:', char_mapping)
        print('Word Segment type mapping:', word_mapping)

        processor = DataProcessor(char_mapping=char_mapping,
                                  word_segment_mapping=word_mapping,
                                  bmes_mapping=BMESToIdMapping())

        x, y = processor.parse_one('одуматься	о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX')
        self.assertEqual(len(x), len(y))
        print(x, y)
        x, y = processor.parse_one('одуматься	о/д/у/м/а/ть/ся')
        print(x, y)

    def test_no_segment_type_processing(self):
        char_mapping = CharToIdMapping(text='одуматься', include_unknown=True)
        word_mapping = WordSegmentTypeToIdMapping(segments=[])

        print('Char mapping:', char_mapping)
        print('Word Segment type mapping:', word_mapping)

        processor = DataProcessor(char_mapping=char_mapping,
                                  word_segment_mapping=word_mapping,
                                  bmes_mapping=BMESToIdMapping())

        x, y = processor.parse_one('-одуматься	-о/д/у/м/ать/ся')
        print(x, y)
        self.assertListEqual(list(x), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(list(y), [0, 2, 3, 3, 3, 0, 1, 2, 0, 2])
