from unittest import TestCase

from src.models.cnn import CNNModel


class TestCNN(TestCase):
    def test_model_structure(self):
        model = CNNModel(nb_symbols=37,
                         embeddings_size=8,
                         dropout=0.2,
                         dense_output_units=64,
                         nb_classes=25)

        model.summary()
