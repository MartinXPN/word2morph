from unittest import TestCase

from src.models.cnn import CNNModel


class TestCNN(TestCase):
    def test_model_structure(self):
        model = CNNModel(nb_symbols=27,
                         embeddings_size=8)

        model.summary()
