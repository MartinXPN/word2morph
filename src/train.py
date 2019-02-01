import os
import random
from typing import Tuple

import fire
import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from src.data.datasets import BucketDataset, Dataset
from src.data.generators import DataGenerator
from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping
from src.data.processing import DataProcessor
from src.models.cnn import CNNModel


class Gym(object):
    def __init__(self):
        self.model: Model = None
        self.train_dataset: Dataset = None
        self.valid_dataset: Dataset = None
        self.char_mapping: CharToIdMapping = None
        self.word_mapping: WordSegmentTypeToIdMapping = None
        self.bmes_mapping: BMESToIdMapping = None
        self.processor: DataProcessor = None

    def fix_random_seed(self, seed: int):
        """ Fix random seed for reproducibility """
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow
            tensorflow.set_random_seed(seed)
        except ImportError:
            pass
        return self

    def init_data(self, train_path: str = 'datasets/rus.train', valid_path: str = 'datasets/rus.dev'):
        self.train_dataset = BucketDataset(file_path=train_path)
        self.valid_dataset = BucketDataset(file_path=valid_path)

        self.bmes_mapping = BMESToIdMapping()
        self.char_mapping = CharToIdMapping(chars=list(self.train_dataset.get_chars()), include_unknown=True)
        self.word_mapping = WordSegmentTypeToIdMapping(segments=self.train_dataset.get_segment_types(),
                                                       include_unknown=False)
        print('Char mapping:', self.char_mapping)
        print('Word Segment type mapping:', self.word_mapping)

        self.processor = DataProcessor(char_mapping=self.char_mapping,
                                       word_segment_mapping=self.word_mapping,
                                       bmes_mapping=self.bmes_mapping)
        return self

    def construct_model(self, embeddings_size: int=8,
                        kernel_sizes: Tuple[int, ...]=(5, 5, 5),
                        nb_filters: Tuple[int, ...]=(192, 192, 192),
                        dense_output_units: int=64,
                        dropout: float=0.2):
        self.model = CNNModel(nb_symbols=len(self.char_mapping),
                              embeddings_size=embeddings_size,
                              kernel_sizes=kernel_sizes,
                              nb_filters=nb_filters,
                              dense_output_units=dense_output_units,
                              dropout=dropout,
                              nb_classes=self.processor.nb_classes())

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()
        return self

    def train(self,
              batch_size: int = 32, epochs: int = 100, patience: int = 10,
              log_dir: str = 'checkpoints', models_dir: str = 'checkpoints'):

        self.model.fit_generator(
            generator=DataGenerator(dataset=self.train_dataset, processor=self.processor, batch_size=batch_size),
            validation_data=DataGenerator(dataset=self.valid_dataset, processor=self.processor, batch_size=batch_size),
            validation_steps=100,
            steps_per_epoch=len(self.train_dataset) // batch_size, epochs=epochs,
            callbacks=[  # AllMetrics(valid_data[:-1], valid_data[-1]),
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=os.path.join(models_dir, 'model-{epoch:02d}-loss-{val_loss:.2f}.hdf5'),
                                monitor='val_loss', save_best_only=True, verbose=1, mode='max'),
                EarlyStopping(patience=patience)
            ],
            # class_weight=class_weights,
        )


if __name__ == '__main__':
    fire.Fire(Gym)
