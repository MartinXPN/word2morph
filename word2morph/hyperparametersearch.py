from copy import deepcopy

import fire
import numpy as np
from btb import HyperParameter, ParamTypes
from btb.selection import UCB1
from btb.tuning import GP

from word2morph.train import Gym
from word2morph.util.args import map_arguments


class HyperparameterSearchGym(Gym):
    def __init__(self):
        super(HyperparameterSearchGym, self).__init__()

        generic_params = [
            ('embeddings_size',         HyperParameter(ParamTypes.INT, [4, 18])),
            ('dense_output_units',      HyperParameter(ParamTypes.INT, [16, 256])),
            ('batch_size',              HyperParameter(ParamTypes.INT, [4, 128])),
            ('dropout',                 HyperParameter(ParamTypes.FLOAT, [0., 0.6])),
            ('use_crf',                 HyperParameter(ParamTypes.BOOL, [True, False])),
        ]

        ''' CNN Models '''
        cnn3 = [
                   ('kernel_sizes-1',   HyperParameter(ParamTypes.INT, [3, 7])),
                   ('kernel_sizes-2',   HyperParameter(ParamTypes.INT, [3, 7])),
                   ('kernel_sizes-3',   HyperParameter(ParamTypes.INT, [3, 7])),
                   ('nb_filters-1',     HyperParameter(ParamTypes.INT, [32, 384])),
                   ('nb_filters-2',     HyperParameter(ParamTypes.INT, [32, 384])),
                   ('nb_filters-3',     HyperParameter(ParamTypes.INT, [32, 384])),
                   ('dilations-1',      HyperParameter(ParamTypes.INT, [1, 5])),
                   ('dilations-2',      HyperParameter(ParamTypes.INT, [1, 5])),
                   ('dilations-3',      HyperParameter(ParamTypes.INT, [1, 5])),
               ] + deepcopy(generic_params)
        cnn4 = [
            ('kernel_sizes-4',          HyperParameter(ParamTypes.INT, [3, 7])),
            ('nb_filters-4',            HyperParameter(ParamTypes.INT, [32, 384])),
            ('dilations-4',             HyperParameter(ParamTypes.INT, [1, 5])),
           ] + deepcopy(cnn3)

        ''' RNN Models '''
        rnn2 = [
            ('recurrent_units-1',       HyperParameter(ParamTypes.INT, [16, 512])),
            ('recurrent_units-2',       HyperParameter(ParamTypes.INT, [16, 512])),
        ] + deepcopy(generic_params)
        rnn3 = [
            ('recurrent_units-3',       HyperParameter(ParamTypes.INT, [16, 512])),
        ] + deepcopy(rnn2)

        self.tuners = {
            'CNN-3': GP(cnn3),
            'CNN-4': GP(cnn4),
            'RNN-2': GP(rnn2),
            'RNN-3': GP(rnn3),
        }
        self.selector = UCB1(list(self.tuners.keys()))

    def search_hyperparameters(self, nb_trials: int, epochs: int = 100, patience: int = 10,
                               early_stop_if_lower_than_best: bool = False,
                               monitor_metric: str = 'val_word_acc_processed', log_dir: str = 'logs'):
        best_training_curve = {key: [0] * epochs for key in self.tuners.keys()}
        for trial in range(nb_trials):
            model_choice = self.selector.select({name: self.tuners[name].y for name in self.tuners.keys()})
            parameters = self.tuners[model_choice].propose()

            model_name, model_depth = model_choice.split('-')
            model_depth = int(model_depth)

            ''' transform parameters '''
            transformed_params = {}
            for key, value in parameters.items():
                if isinstance(value, np.ndarray):
                    value = tuple(value.tolist())
                if isinstance(value, np.generic):
                    value = value.item()

                if '-' in key:
                    feature_name, feature_depth = key.split('-')
                    feature_depth = int(feature_depth) - 1
                    if feature_name not in transformed_params:
                        transformed_params[feature_name] = [None] * model_depth
                    transformed_params[feature_name][feature_depth] = value
                else:
                    transformed_params[key] = value

            transformed_params.update(model_type=model_name,
                                      epochs=epochs, patience=patience, log_dir=log_dir,
                                      monitor_metric=monitor_metric,
                                      best_training_curve=best_training_curve[model_choice]
                                      if early_stop_if_lower_than_best else None,
                                      save_best=False)

            ''' Construct and train the model '''
            print(f'\n\n\nTraining the model: {model_choice} with hyperparameters: {transformed_params}')
            self.construct_model(**map_arguments(self.construct_model, transformed_params))
            history = self.train(**map_arguments(self.train, transformed_params))

            ''' Track results '''
            best_epoch = np.argmax(history[monitor_metric])
            best_score = history[monitor_metric][best_epoch]
            best_score_so_far = np.max(best_training_curve[model_choice])
            potential_improvement = max(0, best_training_curve[model_choice][best_epoch] - best_score)\
                if len(best_training_curve[model_choice]) > best_epoch else 0

            if best_score_so_far < best_score:
                best_training_curve[model_choice] = history[monitor_metric]
            self.tuners[model_choice].add(parameters, best_score + potential_improvement)

        for model_choice in self.tuners.keys():
            model = self.tuners[model_choice]
            # noinspection PyProtectedMember
            print(f'Best score for {model_choice} model: {model._best_score} '
                  f'with hyperparameters: {model._best_hyperparams}')


if __name__ == '__main__':
    fire.Fire(HyperparameterSearchGym)
