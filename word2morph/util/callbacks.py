from typing import List, Optional

from keras.callbacks import EarlyStopping


class ComparableEarlyStopping(EarlyStopping):
    def __init__(self, to_compare_values: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.compare_values = to_compare_values
        print('Compare values:', self.compare_values)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or self.compare_values is None:
            return

        ''' Compare the obtained value to the ones provided in the constructor '''
        if len(self.compare_values) <= epoch or self.monitor_op(current - self.min_delta, self.compare_values[epoch]):
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
