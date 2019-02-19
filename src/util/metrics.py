from pprint import pprint
from typing import Tuple, List

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             accuracy_score, log_loss, roc_auc_score)
from sklearn.preprocessing import LabelBinarizer

from src.data.generators import DataGenerator


def multi_class_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


class Evaluate(Callback):
    # TODO -> evaluate before correcting the predictions and evaluate after to see the difference
    def __init__(self, data_generator: DataGenerator):
        super(Evaluate, self).__init__()
        self.data_generator = data_generator

    def evaluate(self,
                 predictions: List[np.ndarray],
                 labels: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[Tuple[str, float], ...]]:
        """
        Calculates:
         * word-level accuracy
         * char-level metrics: acc, loss, precision, recall, f1, auc, confusion matrix
        :return confusion_matrix, (word_acc, acc, loss, precision, recall, f1, auc)
        """

        ''' Calculate word-level accuracy '''
        correct = 0
        nb_words = 0
        for batch_prediction, batch_label in zip(predictions, labels):
            correct += sum([np.array_equal(np.argmax(word_prediction, axis=1), np.argmax(word_label, axis=1))
                            for word_prediction, word_label in zip(batch_prediction, batch_label)])
            nb_words += len(batch_label)

        ''' Calculate char-level metrics '''
        char_predictions = []
        char_labels = []
        for batch_prediction, batch_label in zip(predictions, labels):
            for word_prediction, word_label in zip(batch_prediction, batch_label):
                char_predictions += word_prediction.tolist()
                char_labels += word_label.tolist()

        char_predictions = np.array(char_predictions)
        char_labels = np.array(char_labels)

        t, p = np.argmax(char_labels, axis=1), np.argmax(char_predictions, axis=1)
        return confusion_matrix(t, p), tuple([('word_acc', correct / nb_words),
                                              ('acc', accuracy_score(t, p)),
                                              ('loss', log_loss(char_labels, char_predictions)),
                                              ('precision', precision_score(t, p, average='macro')),
                                              ('recall', recall_score(t, p, average='macro')),
                                              ('f1', f1_score(t, p, average='macro')),
                                              ('auc', multi_class_roc_auc_score(t, p))])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        epoch_labels = []
        epoch_predictions = []
        for i in range(len(self.data_generator)):
            inputs, labels = next(self.data_generator)
            predictions = self.model.predict(inputs)
            epoch_labels.append(labels)
            epoch_predictions.append(predictions)

        cf_matrix, metrics = self.evaluate(predictions=epoch_predictions, labels=epoch_labels)
        for metric_name, metric_value in metrics:
            logs['val_' + metric_name] = metric_value

        print('\nEvaluating for epoch {}...'.format(epoch + 1))
        print('Confusion Matrix:\n', cf_matrix)
        pprint(logs)
