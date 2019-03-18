import warnings
from typing import List, Optional

from keras.callbacks import EarlyStopping, ModelCheckpoint


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


class Checkpoint(ModelCheckpoint):
    def __init__(self, on_save_callback, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        super().__init__('', monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.on_save = on_save_callback

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.on_save()
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.on_save()
