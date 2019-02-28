import gc
import itertools
import os
import random
import sys
from datetime import datetime
from itertools import chain
from typing import Tuple, Dict, List, Optional

import fire
import numpy as np
from keras import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.utils import class_weight

from src.data.generators import DataGenerator
from src.data.loaders import DataLoader
from src.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping, LabelToIdMapping
from src.data.processing import DataProcessor
from src.entities.dataset import BucketDataset, Dataset
from src.models.cnn import CNNModel
from src.models.rnn import RNNModel
from src.util.args import map_arguments
from src.util.filesystem import save_file
from src.util.heuristics import ComparableEarlyStopping
from src.util.metrics import Evaluate


class Gym(object):
    def __init__(self):
        self.model: Model = None
        self.train_dataset: Dataset = None
        self.valid_dataset: Dataset = None
        self.char_mapping: CharToIdMapping = None
        self.word_mapping: WordSegmentTypeToIdMapping = None
        self.bmes_mapping: BMESToIdMapping = None
        self.label_mapping: LabelToIdMapping = None
        self.processor: DataProcessor = None
        self.class_weights: Dict = None
        self.params: Dict = {}

    def fix_random_seed(self, seed: int):
        """ Fix random seed for reproducibility """
        self.params.update(locals())
        random.seed(seed)
        np.random.seed(seed)
        try:
            # noinspection PyUnresolvedReferences
            import tensorflow
            tensorflow.set_random_seed(seed)
        except ImportError:
            pass
        return self

    def init_data(self, train_path: str = 'datasets/rus.train', valid_path: str = 'datasets/rus.test'):
        self.params.update(locals())
        self.train_dataset = BucketDataset(samples=DataLoader(file_path=train_path).load())
        self.valid_dataset = BucketDataset(samples=DataLoader(file_path=valid_path).load())

        self.bmes_mapping = BMESToIdMapping()
        self.char_mapping = CharToIdMapping(chars=list(self.train_dataset.get_chars()), include_unknown=True)
        self.word_mapping = WordSegmentTypeToIdMapping(segments=self.train_dataset.get_segment_types(),
                                                       include_unknown=False)
        print('Char mapping:', self.char_mapping)
        print('Word Segment type mapping:', self.word_mapping)
        print('BMES mapping:', self.bmes_mapping)

        self.processor = DataProcessor(char_mapping=self.char_mapping,
                                       word_segment_mapping=self.word_mapping,
                                       bmes_mapping=self.bmes_mapping)

        print('Removing wrong labels (current labels are: the cross product [BMES x SegmentTypes])...')
        labels = list(chain.from_iterable([self.processor.parse_one(sample)[1] for sample in self.train_dataset]))
        labels = np.array(labels, dtype=np.int8)
        unique_labels = np.unique(labels)

        self.label_mapping = LabelToIdMapping(labels=list(unique_labels))
        self.class_weights = class_weight.compute_class_weight('balanced', unique_labels, labels)
        self.processor.label_mapping = self.label_mapping
        print('Calculated class weights:', self.class_weights)
        print('Number of classes per char:', self.processor.nb_classes())
        return self

    def construct_model(self,
                        model_type: str = 'CNN',
                        embeddings_size: int=8,
                        kernel_sizes: Tuple[int, ...]=(5, 5, 5),
                        nb_filters: Tuple[int, ...]=(192, 192, 192),
                        recurrent_units: Tuple[int, ...] = (64, 128, 256),
                        dense_output_units: int=64,
                        dropout: float=0.2):
        self.params.update(locals())

        # Clean-up the keras session before constructing a new model
        del self.model
        K.clear_session()
        gc.collect()

        model_type = model_type.upper()
        if model_type == 'CNN':
            self.model = CNNModel(nb_symbols=len(self.char_mapping),
                                  embeddings_size=embeddings_size,
                                  kernel_sizes=kernel_sizes,
                                  nb_filters=nb_filters,
                                  dense_output_units=dense_output_units,
                                  dropout=dropout,
                                  nb_classes=self.processor.nb_classes())
        elif model_type == 'RNN':
            self.model = RNNModel(nb_symbols=len(self.char_mapping),
                                  embeddings_size=embeddings_size,
                                  recurrent_units=recurrent_units,
                                  dense_output_units=dense_output_units,
                                  dropout=dropout,
                                  nb_classes=self.processor.nb_classes())
        else:
            raise ValueError('Cannot find implementation for the model type {}'.format(model_type))

        self.model.compile(optimizer=Adam(clipnorm=5.0), loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()
        return self

    def train(self, batch_size: int = 32, epochs: int = 100, patience: int = 10,
              best_training_curve: Optional[List[float]] = None,
              monitor_metric: str = 'val_acc', log_dir: str = 'logs'):
        self.params.update(locals()), self.params.pop('self'), self.params.pop('best_training_curve')

        ''' Save all the objects/parameters for reproducibility '''
        log_dir = os.path.join(log_dir, datetime.now().replace(microsecond=0).isoformat())
        models_dir = os.path.join(log_dir, 'checkpoints/')
        os.makedirs(models_dir)
        save_file(os.path.join(log_dir, 'commandline.txt'), obj=sys.argv, save_pickled=False)
        save_file(os.path.join(log_dir, 'params.txt'), obj=self.params, save_pickled=False)
        save_file(os.path.join(log_dir, 'processor.pkl'), obj=self.processor, save_pickled=True)

        train_generator = DataGenerator(dataset=self.train_dataset, processor=self.processor, batch_size=batch_size)
        valid_generator = DataGenerator(dataset=self.valid_dataset, processor=self.processor, batch_size=batch_size)

        history = self.model.fit_generator(
            generator=itertools.cycle(train_generator),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[Evaluate(data_generator=itertools.cycle(valid_generator), nb_steps=len(valid_generator)),
                       TensorBoard(log_dir=log_dir),
                       ModelCheckpoint(filepath=os.path.join(models_dir, '{epoch:02d}-loss-{val_loss:.2f}.hdf5'),
                                       monitor=monitor_metric, save_best_only=True, verbose=1),
                       ComparableEarlyStopping(to_compare_values=best_training_curve, monitor=monitor_metric, patience=patience)],
            class_weight=self.class_weights,
        )
        return history.history

    def run(self, **kwargs: Dict):
        self.fix_random_seed(**map_arguments(self.fix_random_seed, kwargs)) \
            .init_data(**map_arguments(self.init_data, kwargs)) \
            .construct_model(**map_arguments(self.construct_model, kwargs)) \
            .train(**map_arguments(self.train, kwargs))


if __name__ == '__main__':
    fire.Fire(Gym)
