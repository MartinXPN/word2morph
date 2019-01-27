from unittest import TestCase

from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping
from src.data.processing import DataProcessor


class TestPreprocessing(TestCase):
    def setUp(self):
        char_mapping = CharToIdMapping('одуматься', include_unknown=True)
        word_mapping = WordSegmentTypeToIdMapping(segments=['PREF', 'ROOT', 'SUFF'])

        self.processor = DataProcessor(char_mapping=char_mapping,
                                       word_segment_mapping=word_mapping,
                                       bmes_mapping=BMESToIdMapping())

    def test_processing(self):
        x, y = self.processor.parse_one('одуматься	о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX')
        print(x, y)
        x, y = self.processor.parse_one('одуматься	о/д/у/м/а/ть/ся')
        print(x, y)
