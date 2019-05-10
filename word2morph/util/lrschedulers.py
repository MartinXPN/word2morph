import math


class ExponentialDecay:
    def __init__(self, initial_lr: float, rate: float = 0.1):
        self.initial_lr = initial_lr
        self.rate = rate

    def __call__(self, epoch, *args, **kwargs):
        return self.initial_lr * math.exp(-self.rate * epoch)


class StepDecay:
    def __init__(self, initial_lr: float, drop: float = 0.5, interval: float = 10.):
        self.initial_lr = initial_lr
        self.drop = drop
        self.interval = interval

    def __call__(self, epoch, *args, **kwargs):
        drop_magnitude = math.floor((1 + epoch) / self.interval)
        return self.initial_lr * math.pow(self.drop, drop_magnitude)


class Constant:
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr

    def __call__(self, epoch, *args, **kwargs):
        return self.initial_lr
