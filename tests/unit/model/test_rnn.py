from unittest import TestCase

import numpy as np
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from word2morph.data.loaders import DataLoader
from word2morph.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping
from word2morph.data.processing import DataProcessor
from word2morph.models.rnn import RNNModel


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
                         use_crf=False,
                         nb_classes=25)

        model.summary()
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

        loader = DataLoader(samples=['одуматься	о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX'])
        x, y = self.processor.parse_one(sample=loader.load()[0])
        pred = model.predict(np.array([x]))
        self.assertEqual(pred.shape, (1, 9, 25))
        print(x.shape)
        print(y.shape)

    def test_crf_model(self):
        model = RNNModel(nb_symbols=37,
                         embeddings_size=8,
                         dropout=0.2,
                         use_crf=True,
                         nb_classes=25)

        model.summary()
        model.compile('adam', crf_loss, metrics=[crf_viterbi_accuracy])

        loader = DataLoader(samples=['одуматься	о:PREF/дум:ROOT/а:SUFF/ть:SUFF/ся:POSTFIX'])
        x, y = self.processor.parse_one(sample=loader.load()[0])
        pred = model.predict(np.array([x]))
        self.assertEqual(pred.shape, (1, 9, 25))
        print('CRF output:', pred)
        print(x.shape)
        print(y.shape)
