import itertools
import json
import sys
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import fire
import gc
import numpy as np
from keras import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from sklearn.utils import class_weight

from word2morph import Word2Morph
from word2morph.data.generators import DataGenerator
from word2morph.data.loaders import DataLoader
from word2morph.data.mappings import CharToIdMapping, WordSegmentTypeToIdMapping, BMESToIdMapping, LabelToIdMapping
from word2morph.data.processing import DataProcessor
from word2morph.entities.dataset import BucketDataset, Dataset
from word2morph.models.cnn import CNNModel
from word2morph.models.rnn import RNNModel
from word2morph.util.args import map_arguments
from word2morph.util.callbacks import ComparableEarlyStopping, Checkpoint, ClassifierTensorBoard
from word2morph.util.metrics import Evaluate


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
                        embeddings_size: int = 8,
                        kernel_sizes: Tuple[int, ...] = (5, 5, 5),
                        nb_filters: Tuple[int, ...] = (192, 192, 192),
                        recurrent_units: Tuple[int, ...] = (64, 128, 256),
                        dense_output_units: int = 64,
                        use_crf: bool = True,
                        dropout: float = 0.2):
        self.params.update(locals())

        # Clean-up the keras session before constructing a new model
        del self.model
        K.clear_session()
        gc.collect()

        model_type = model_type.upper()
        if model_type == 'CNN':
            self.model = CNNModel(nb_symbols=len(self.char_mapping), embeddings_size=embeddings_size,
                                  kernel_sizes=kernel_sizes, nb_filters=nb_filters,
                                  dense_output_units=dense_output_units,
                                  dropout=dropout, use_crf=use_crf,
                                  nb_classes=self.processor.nb_classes())
        elif model_type == 'RNN':
            self.model = RNNModel(nb_symbols=len(self.char_mapping), embeddings_size=embeddings_size,
                                  recurrent_units=recurrent_units,
                                  dense_output_units=dense_output_units,
                                  dropout=dropout, use_crf=use_crf,
                                  nb_classes=self.processor.nb_classes())
        else:
            raise ValueError(f'Cannot find implementation for the model type {model_type}')

        loss = crf_loss if use_crf else 'categorical_crossentropy'
        metrics = [crf_viterbi_accuracy] if use_crf else ['acc']
        self.model.compile(optimizer=Adam(clipnorm=5.0), loss=loss, metrics=metrics)
        self.model.summary()
        return self

    def train(self, batch_size: int = 32, epochs: int = 100, patience: int = 10,
              best_training_curve: Optional[List[float]] = None, save_best: bool = True,
              monitor_metric: str = 'val_acc', log_dir: str = 'logs'):
        self.params.update(locals()), self.params.pop('self'), self.params.pop('best_training_curve')

        ''' Save all the objects/parameters for reproducibility '''
        log_dir = Path(log_dir).joinpath(datetime.now().replace(microsecond=0).isoformat())
        model_path = Path(log_dir).joinpath('checkpoints').joinpath("best-model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(fname=Path(log_dir).joinpath('commandline.txt'), X=sys.argv, fmt='%s')
        np.savetxt(fname=Path(log_dir).joinpath('params.txt'), X=[json.dumps(self.params, indent=4, sort_keys=True)], fmt='%s')

        train_generator = DataGenerator(dataset=self.train_dataset, processor=self.processor, batch_size=batch_size)
        valid_generator = DataGenerator(dataset=self.valid_dataset, processor=self.processor, batch_size=batch_size)

        history = self.model.fit_generator(
            generator=itertools.cycle(train_generator),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[Evaluate(data_generator=itertools.cycle(valid_generator), nb_steps=len(valid_generator)),
                       ClassifierTensorBoard(labels=[f'{k}: {i}' for k, i in self.label_mapping.key_to_id.items()], log_dir=log_dir),
                       Checkpoint(on_save_callback=(lambda: Word2Morph(model=self.model, processor=self.processor).save(model_path)) if save_best else lambda: None, monitor=monitor_metric, save_best_only=True, verbose=1),
                       ComparableEarlyStopping(to_compare_values=best_training_curve, monitor=monitor_metric, patience=patience),
                       EarlyStopping(monitor=monitor_metric, patience=patience)],
            class_weight=self.class_weights,
        )

        return history.history

    def run(self, **kwargs: Dict):
        self.init_data(**map_arguments(self.init_data, kwargs)) \
            .construct_model(**map_arguments(self.construct_model, kwargs)) \
            .train(**map_arguments(self.train, kwargs))


if __name__ == '__main__':
    fire.Fire(Gym)
