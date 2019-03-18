from typing import List

from keras.callbacks import EarlyStopping


class ComparableEarlyStopping(EarlyStopping):
    def __init__(self, to_compare_values: List[float], **kwargs):
        super().__init__(**kwargs)
        self.compare_values = to_compare_values
        print('Compare values:', self.compare_values)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        ''' Compare the obtained value to the ones provided in the constructor '''
        if len(self.compare_values) <= epoch or self.monitor_op(current - self.min_delta, self.compare_values[epoch]):
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of '
                          'the best epoch')
                self.model.set_weights(self.best_weights)
