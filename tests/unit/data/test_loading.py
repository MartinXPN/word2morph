from unittest import TestCase

from src.data.loaders import DataLoader


class TestSampleDataset(TestCase):
    def test_dataset(self):
        loader = DataLoader(samples=['accompanied	ac/compani/ed',
                                     'acknowledging	ac/knowledg/ing',
                                     'defections	defect/ion/s'])
        loaded_samples = loader.load()
        self.assertEqual(len(loaded_samples), 3)
        self.assertEqual(loaded_samples[0].word, 'accompanied')
        self.assertEqual(loaded_samples[1].word, 'acknowledging')
        self.assertEqual(loaded_samples[1].segments[1].segment, 'knowledg')
        self.assertIsNone(loaded_samples[1].segments[1].type)
