from unittest import TestCase

from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping
from src.data.processing import DataProcessor
from src.entities.sample import Sample, Segment


class TestPreProcessing(TestCase):

    def test_processing(self):
        char_mapping = CharToIdMapping(text='одуматься', include_unknown=True)
        word_mapping = WordSegmentTypeToIdMapping(segments=['PREF', 'ROOT', 'SUFF'])

        print('Char mapping:', char_mapping)
        print('Word Segment type mapping:', word_mapping)

        processor = DataProcessor(char_mapping=char_mapping,
                                  word_segment_mapping=word_mapping,
                                  bmes_mapping=BMESToIdMapping())

        # 'одуматься	о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX'
        s = Sample(word='одуматься', segments=(Segment(segment='о', segment_type='PREF'),
                                               Segment(segment='дум', segment_type='ROOT'),
                                               Segment(segment='а', segment_type='SUFF'),
                                               Segment(segment='ть', segment_type='SUFF'),
                                               Segment(segment='ся', segment_type='POSTFIX')))
        x, y = processor.parse_one(sample=s)
        self.assertEqual(len(x), len(y))
        print(x, y)

    def test_no_segment_type_processing(self):
        char_mapping = CharToIdMapping(text='одуматься', include_unknown=True)
        word_mapping = WordSegmentTypeToIdMapping(segments=[])

        print('Char mapping:', char_mapping)
        print('Word Segment type mapping:', word_mapping)

        processor = DataProcessor(char_mapping=char_mapping,
                                  word_segment_mapping=word_mapping,
                                  bmes_mapping=BMESToIdMapping())

        s = Sample(word='-одуматься', segments=(Segment(segment='-о'),
                                                Segment(segment='д'),
                                                Segment(segment='у'),
                                                Segment(segment='м'),
                                                Segment(segment='ать'),
                                                Segment(segment='ся')))

        x, y = processor.parse_one(sample=s)
        print(x, y)
        self.assertListEqual(list(x), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(list(y), [0, 2, 3, 3, 3, 0, 1, 2, 0, 2])


class TestPostProcessing(TestCase):
    def test_prediction_to_sample(self):
        pass
