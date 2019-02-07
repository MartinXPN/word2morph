import numpy as np

import fire
from btb.hyper_parameter import CatHyperParameter
from btb.tuning import GP
from btb import HyperParameter, ParamTypes
from btb.selection import UCB1

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
                ('embeddings_size',     HyperParameter(ParamTypes.INT, [4, 64])),
                ('kernel_sizes',        TupleHyperparameter(param_range=[(7, 7, 7), (5, 5, 5), (3, 3, 3)])),
                ('nb_filters',          TupleHyperparameter(param_range=[(192, 192, 192), (128, 128, 128)])),
                ('dense_output_units',  HyperParameter(ParamTypes.INT, [16, 256])),
                ('dropout',             HyperParameter(ParamTypes.FLOAT, [0., 0.6])),
                ('batch_size',          HyperParameter(ParamTypes.INT, [4, 512])),
            ]),
            # 'RNN': GP([
            #     ('embeddings_size',     HyperParameter(ParamTypes.INT, [4, 64])),
            #     ('dropout',             HyperParameter(ParamTypes.FLOAT, [0., 0.6])),
            # ])
        }
        self.selector = UCB1(list(self.tuners.keys()))

    def search_hyperparameters(self, nb_trials: int, epochs: int = 100, patience: int = 10,
                               log_dir: str = 'logs', models_dir: str = 'checkpoints'):
        for trial in range(nb_trials):
            next_choice = self.selector.select({'CNN': self.tuners['CNN'].y})
            parameters = self.tuners[next_choice].propose()
            transformed_params = {key: tuple(i.item() for i in value)
                                  if type(value) is np.ndarray else value
                                  for (key, value) in parameters.items()}
            print('\n\n\nTraining the model: {} with hyperparameters: {}'.format(next_choice, transformed_params))

            ''' Construct and train the model with the selected parameters '''
            self.construct_model(**map_arguments(self.construct_model, transformed_params))
            transformed_params.update(epochs=epochs, patience=patience, log_dir=log_dir, models_dir=models_dir)
            history = self.train(**map_arguments(self.train, transformed_params))
            self.tuners[next_choice].add(parameters, history['val_acc'][-1])


if __name__ == '__main__':
    fire.Fire(HyperparameterSearchGym)
