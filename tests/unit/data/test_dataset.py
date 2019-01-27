import os
from unittest import TestCase

from src.data.datasets import Dataset


class TestDataset(TestCase):
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
