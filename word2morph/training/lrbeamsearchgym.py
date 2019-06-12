import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import gc
from keras import backend as K

from word2morph import Word2Morph
from word2morph.data.generators import DataGenerator
from word2morph.training.gym import Gym
from word2morph.util.metrics import Evaluate
from word2morph.util.utils import get_current_commit


class ModelInstance:
    def __init__(self, performance, path, lr):
        self.performance = performance
        self.path = path
        self.lr = lr

    def __eq__(self, other: 'ModelInstance'):
        return self.performance, self.path, self.lr == other.performance, other.path, other.lr

    def __lt__(self, other: 'ModelInstance'):
        return self.performance < other.performance

    def __str__(self):
        return f'Model with perf: {self.performance}, lr: {self.lr}, saved at: {self.path}'

    def __hash__(self):
        return hash(self.path)


class LearningRateBeamSearchGym(Gym):

    def train(self, batch_size: int = 32, epochs: int = 100,
              lr_multipliers: Tuple[float, ...] = (0.5, 0.75, 0.8, 1, 1.2, 1.5, 2), nb_models: int = 3,
              threads: int = 4,
              monitor_metric: str = 'val_word_acc_processed', log_dir: str = 'logs',
              **kwargs):
        self.params.update(locals()), self.params.pop('self')

        ''' Save all the objects/parameters for reproducibility '''
        log_dir = Path(log_dir).joinpath(datetime.now().replace(microsecond=0).isoformat())
        model_path = Path(log_dir).joinpath('checkpoints').joinpath("best-model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(log_dir).joinpath('params.json'), 'w', encoding='utf-8') as f:
            json.dump({'params': self.params, 'commandline': sys.argv, 'commit': get_current_commit()}, f, indent=4)

        train_generator = DataGenerator(dataset=self.train_dataset, processor=self.processor, batch_size=batch_size)
        valid_generator = DataGenerator(dataset=self.valid_dataset, processor=self.processor, batch_size=batch_size,
                                        with_samples=True)

        best_current_models: List[ModelInstance] = []
        best_prev_models: List[ModelInstance] = []

        for epoch in range(epochs):
            best_prev_models = deepcopy(best_current_models)
            best_current_models = []

            def log_model(score):
                nonlocal best_current_models
                learning_rate = float(K.get_value(self.model.optimizer.lr))
                path = f'{log_dir}/model-epoch:{epoch}-acc:{score:.3f}-lr:{learning_rate:.3f}.joblib'
                best_current_models.append(ModelInstance(performance=score, path=path, lr=learning_rate))
                print('Obtained:', str(best_current_models[-1]), flush=True)
                Word2Morph(model=self.model, processor=self.processor).save(path)

                best_current_models = list(set(best_current_models))
                best_current_models = sorted(best_current_models, reverse=True)
                best_current_models, worst = best_current_models[:nb_models], best_current_models[nb_models:]
                for model in worst:
                    print('Removing:', model.path, flush=True)
                    os.remove(model.path)

                print('Resulting list:')
                for i, model in enumerate(best_current_models):
                    print(i, ':', str(model))
                print(flush=True)

            # There are no models for the initial epoch => use the initial random model as the base model
            if len(best_current_models) == 0:
                log_model(score=0)

            for base_model in best_prev_models:
                for lr_multiplier in lr_multipliers:
                    print('Trying to modify:', str(base_model), flush=True)

                    # Clean-up the keras session before working with a new model
                    del self.processor
                    del self.model
                    K.clear_session()
                    gc.collect()

                    w2m = Word2Morph.load_model(base_model.path)
                    self.model, self.processor = w2m.model, w2m.processor
                    lr = float(K.get_value(self.model.optimizer.lr))
                    K.set_value(self.model.optimizer.lr, lr * lr_multiplier)

                    history = self.model.fit_generator(
                        generator=train_generator,
                        epochs=epoch + 1, initial_epoch=epoch,
                        callbacks=[Evaluate(data_generator=valid_generator, to_sample=self.processor.to_sample)],
                        class_weight=self.class_weights,
                        use_multiprocessing=True, workers=threads,
                    )
                    log_model(score=history.history[monitor_metric][-1])
