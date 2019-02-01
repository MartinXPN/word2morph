import os
from unittest import TestCase

from src.data.datasets import Dataset, BucketDataset


class TestFileDataset(TestCase):
    def setUp(self) -> None:
        self.filename = 'test_data.txt'
        with open(self.filename, 'w') as f:
            f.write('accompanied	ac/compani/ed\n')
            f.write('acknowledging	ac/knowledg/ing\n')
            f.write('defections	defect/ion/s\n')

        self.dataset = Dataset(self.filename)

    def test_dataset_size(self):
        self.assertEqual(len(self.dataset), 3)

    def doCleanups(self) -> None:
        os.remove(self.filename)


class TestSampleDataset(TestCase):
    def test_dataset(self):
        dataset = Dataset(samples=['accompanied	ac/compani/ed',
                                   'acknowledging	ac/knowledg/ing',
                                   'defections	defect/ion/s'])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], 'accompanied	ac/compani/ed')


class TestFailDataset(TestCase):
    def test_none_provided(self):
        with self.assertRaises(ValueError):
            _ = Dataset()

    def test_both_provided(self):
        with self.assertRaises(ValueError):
            _ = Dataset(file_path='file.txt', samples=['accompanied	ac/compani/ed'])


class TestBucketDataset(TestCase):
    def test_buckets(self):
        dataset = BucketDataset(samples=['accompanied	ac/compani/ed',
                                         'acknowledging	ac/knowledg/ing',
                                         'abcdowledging	ac/knowledg/ing',
                                         'akpmowledging	ac/knowledg/ing',
                                         'anowledging	ac/knowledg/ing',
                                         'defections	defect/ion/s'])
        print('\n')
        print(dataset.data)
        print([(item[0], [line.split('\t')[0] for line in item[1]]) for item in dataset.buckets.items()])
        print([(item[0], len(item[1])) for item in dataset.buckets.items()])

        before_shuffling_length = len(dataset)
        dataset.shuffle()
        self.assertEqual(len(dataset), before_shuffling_length)

        print([(item[0], len(item[1])) for item in dataset.buckets.items()])
        print([(item[0], [line.split('\t')[0] for line in item[1]]) for item in dataset.buckets.items()])

        self.assertEqual(len(dataset.buckets[13]), 3)
        self.assertEqual(len(dataset.buckets[11]), 2)

        print(dataset.data)
