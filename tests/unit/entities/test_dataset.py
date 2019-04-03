from unittest import TestCase

from word2morph.data.loaders import DataLoader
from word2morph.entities.dataset import Dataset, BucketDataset


class TestSampleDataset(TestCase):
    def test_dataset(self):
        dataset = Dataset(DataLoader(samples=['accompanied\tac/compani/ed',
                                              'acknowledging\tac/knowledg/ing',
                                              'defections\tdefect/ion/s']).load())
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0].word, 'accompanied')
        self.assertEqual(dataset[1].segments[0].segment, 'ac')
        self.assertEqual(dataset[1].segments[1].segment, 'knowledg')


class TestBucketDataset(TestCase):
    def test_buckets(self):
        dataset = BucketDataset(DataLoader(samples=['accompanied\tac/compani/ed',
                                                    'acknowledging\tac/knowledg/ing',
                                                    'abcdowledging\tac/knowledg/ing',
                                                    'akpmowledging\tac/knowledg/ing',
                                                    'anowledging\tac/knowledg/ing',
                                                    'defections\tdefect/ion/s']).load())

        print([(length, [str(sample) for sample in samples]) for length, samples in dataset.buckets.items()])
        print([(item[0], len(item[1])) for item in dataset.buckets.items()])

        before_shuffling_length = len(dataset)
        dataset.shuffle()
        self.assertEqual(len(dataset), before_shuffling_length)

        print([(item[0], len(item[1])) for item in dataset.buckets.items()])
        print([(length, [str(sample) for sample in samples]) for length, samples in dataset.buckets.items()])

        self.assertEqual(len(dataset.buckets[13]), 3)
        self.assertEqual(len(dataset.buckets[11]), 2)
