import io
import itertools
import re
import textwrap
import warnings
from typing import List, Optional

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


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


class ClassifierTensorBoard(TensorBoard):
    def __init__(self,
                 labels: List[str],
                 log_dir: str = './logs', histogram_freq: int = 0, batch_size: int = 32,
                 write_graph: bool = True, write_grads: bool = False, write_images: bool = False,
                 embeddings_freq: int = 0, embeddings_layer_names: Optional[List[str]] = None,
                 embeddings_metadata=None, embeddings_data=None,
                 update_freq: str = 'epoch'):
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, batch_size=batch_size,
                         write_graph=write_graph, write_grads=write_grads, write_images=write_images,
                         embeddings_freq=embeddings_freq, embeddings_layer_names=embeddings_layer_names,
                         embeddings_metadata=embeddings_metadata, embeddings_data=embeddings_data,
                         update_freq=update_freq)
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        confusion_matrix = None
        confusion_matrix_key = None
        for key, value in logs.items():
            if isinstance(value, np.ndarray):
                confusion_matrix_key = key
                confusion_matrix = value
                break

        if confusion_matrix_key is not None:
            del logs[confusion_matrix_key]

        # Log confusion matrix
        index = epoch if self.update_freq == 'epoch' else self.samples_seen
        figure = self._plot_confusion_matrix(confusion_matrix)
        summary = self._figure_to_summary(figure)
        self.writer.add_summary(summary, index)

        # Log primitive data
        super().on_epoch_end(epoch, logs)

        if confusion_matrix_key is not None and confusion_matrix is not None:
            logs[confusion_matrix_key] = confusion_matrix

    def _plot_confusion_matrix(self, cm: np.ndarray):
        """
        :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
    `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
        """
        nb_classes = len(self.labels)

        fig = Figure(figsize=(nb_classes, nb_classes), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(nb_classes), range(nb_classes)):
            ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.',
                    horizontalalignment="center",
                    verticalalignment='center',
                    color="black")

        fig.set_tight_layout(True)
        return fig

    @staticmethod
    def _figure_to_summary(fig: Figure):
        """
        Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
        that can be directly fed into ``Summary.FileWriter``.
        :param fig: A ``matplotlib.figure.Figure`` object.
        :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                 as a image summary.
        """

        # attach a new canvas if not exists
        if fig.canvas is None:
            FigureCanvasAgg(fig)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # get PNG data from the figure
        png_buffer = io.BytesIO()
        fig.canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        summary_image = tf.Summary.Image(height=h, width=w, colorspace=4, encoded_image_string=png_encoded)
        summary = tf.Summary(value=[tf.Summary.Value(tag='ConfusionMatrix', image=summary_image)])
        return summary
