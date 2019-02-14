from unittest import TestCase

import numpy as np

from src.data.loaders import DataLoader
from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping
from src.data.processing import DataProcessor
from src.models.rnn import RNNModel


class TestRNN(TestCase):
    def setUp(self):
        char_mapping = CharToIdMapping(text='одуматься', include_unknown=True)
        word_mapping = WordSegmentTypeToIdMapping(segments=['PREF', 'ROOT', 'SUFF'])

        print('Char mapping:', char_mapping)
        print('Word Segment type mapping:', word_mapping)

        self.processor = DataProcessor(char_mapping=char_mapping,
                                       word_segment_mapping=word_mapping,
                                       bmes_mapping=BMESToIdMapping())

    def test_model_structure(self):
        model = RNNModel(nb_symbols=37,
                         embeddings_size=8,
                         dropout=0.2,
                         dense_output_units=64,
                         nb_classes=25)

        model.summary()
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

        loader = DataLoader(samples=['одуматься	о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX'])
        x, y = self.processor.parse_one(sample=loader.load()[0])
        pred = model.predict(np.array([x]))
        self.assertEqual(pred.shape, (1, 9, 25))
        print(x.shape)
        print(y.shape)
