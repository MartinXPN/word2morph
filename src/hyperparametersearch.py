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
                ('embeddings_size',     HyperParameter(ParamTypes.INT, [4, 32])),
                ('kernel_sizes',        TupleHyperparameter(param_range=[(7, 7, 7, 7),      (5, 5, 5, 5),       (3, 3, 3, 3),       (5, 5, 3, 3),   (7, 5, 5, 3)])),
                ('nb_filters',          TupleHyperparameter(param_range=[(192, 192, 192),   (64, 128, 256),     (32, 64, 128, 256), (64, 64, 128, 128)])),
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
            model_choice = self.selector.select({'CNN': self.tuners['CNN'].y})
            parameters = self.tuners[model_choice].propose()

            ''' Construct and train the model with the selected parameters '''
            transformed_params = {key: tuple(value.tolist()) if isinstance(value, np.ndarray) else value
                                  for key, value in parameters.items()}
            transformed_params.update(epochs=epochs, patience=patience, log_dir=log_dir, models_dir=models_dir)

            print('\n\n\nTraining the model: {} with hyperparameters: {}'.format(model_choice, transformed_params))
            self.construct_model(**map_arguments(self.construct_model, transformed_params))
            history = self.train(**map_arguments(self.train, transformed_params))
            self.tuners[model_choice].add(transformed_params, history['val_acc'][-1])


if __name__ == '__main__':
    fire.Fire(HyperparameterSearchGym)
