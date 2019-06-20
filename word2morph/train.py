import fire

from word2morph.training.gym import Gym
from word2morph.training.hyperparametersearch import HyperparameterSearchGym
from word2morph.training.lrbeamsearchgym import LearningRateBeamSearchGym


if __name__ == '__main__':
    fire.Fire({
        'basic_train': Gym,
        'lr_beam_search': LearningRateBeamSearchGym,
        'hyperparameter_search': HyperparameterSearchGym,
    })
