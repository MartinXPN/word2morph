import fire
import numpy as np
from btb import HyperParameter, ParamTypes
from btb.hyper_parameter import CatHyperParameter
from btb.selection import UCB1
from btb.tuning import GP

from src.train import Gym
from src.util.args import map_arguments


class TupleHyperparameter(CatHyperParameter):
    param_type = 10

    def __new__(cls, param_type=None, param_range=None):
        return object.__new__(cls)

    def cast(self, value):
        return tuple(value) if value is not None else None


class HyperparameterSearchGym(Gym):
    def __init__(self):
        super(HyperparameterSearchGym, self).__init__()

        self.tuners = {
            'CNN': GP([
                ('embeddings_size',     HyperParameter(ParamTypes.INT, [4, 18])),
                ('kernel_sizes',        TupleHyperparameter(param_range=[(7, 7, 7, 7),          (5, 5, 5, 5),           (3, 3, 3, 3),           (5, 5, 3, 3),   (7, 5, 5, 3)])),
                ('nb_filters',          TupleHyperparameter(param_range=[(192, 192, 192),       (232, 232, 232),        (192, 232, 256),        (64, 128, 256),
                                                                         (32, 64, 128, 256),    (64, 64, 128, 128),     (64, 128, 198, 256),    (32, 64, 64, 128)])),
                ('dense_output_units',  HyperParameter(ParamTypes.INT, [16, 256])),
                ('dropout',             HyperParameter(ParamTypes.FLOAT, [0., 0.6])),
                ('batch_size',          HyperParameter(ParamTypes.INT, [4, 128])),
            ]),
            'RNN': GP([
                ('embeddings_size',     HyperParameter(ParamTypes.INT, [4, 18])),
                ('recurrent_units',     TupleHyperparameter(param_range=[(64, 128),     (128, 256),     (256, 512),     (128, 128),     (256, 256),
                                                                         (32, 64, 64),  (32, 64, 128),  (64, 64, 128),  (64, 128, 256), (128, 128, 256)])),
                ('dense_output_units',  HyperParameter(ParamTypes.INT, [16, 256])),
                ('dropout',             HyperParameter(ParamTypes.FLOAT, [0., 0.6])),
                ('batch_size',          HyperParameter(ParamTypes.INT, [4, 128])),
            ])
        }
        self.selector = UCB1(list(self.tuners.keys()))

    def search_hyperparameters(self, nb_trials: int, epochs: int = 100, patience: int = 10,
                               monitor_metric: str = 'val_acc', log_dir: str = 'logs'):
        best_training_curve = {key: None for key in self.tuners.keys()}
        for trial in range(nb_trials):
            model_choice = self.selector.select({
                'CNN': self.tuners['CNN'].y,
                'RNN': self.tuners['RNN'].y,
            })
            parameters = self.tuners[model_choice].propose()

            ''' Construct and train the model with the selected parameters '''
            transformed_params = {key: tuple(value.tolist()) if isinstance(value, np.ndarray) else value
                                  for key, value in parameters.items()}
            transformed_params.update(model_type=model_choice, epochs=epochs, patience=patience, log_dir=log_dir,
                                      monitor_metric=monitor_metric,
                                      best_training_curve=best_training_curve[model_choice])

            print('\n\n\nTraining the model: {} with hyperparameters: {}'.format(model_choice, transformed_params))
            self.construct_model(**map_arguments(self.construct_model, transformed_params))
            history = self.train(**map_arguments(self.train, transformed_params))

            best_score = max(history[monitor_metric])
            # noinspection PyProtectedMember
            if self.tuners[model_choice]._best_score < best_score:
                best_training_curve[model_choice] = history[monitor_metric]
            self.tuners[model_choice].add(transformed_params, best_score)

        for model_choice in self.tuners.keys():
            model = self.tuners[model_choice]
            # noinspection PyProtectedMember
            print('Best score for {} model: {} with hyperparameters: {}'.format(model_choice,
                                                                                model._best_score,
                                                                                model._best_hyperparams))


if __name__ == '__main__':
    fire.Fire(HyperparameterSearchGym)
