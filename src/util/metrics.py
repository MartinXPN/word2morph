import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, log_loss


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))


class AllMetrics(Callback):
    def __init__(self, inputs, labels):
        super(AllMetrics, self).__init__()
        self.inputs = inputs
        self.labels = labels
        self.accuracy = None
        self.loss = None
        self.confusion_matrix = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predictions = self.model.predict(self.inputs)
        t, p = np.argmax(self.labels, axis=1), np.argmax(predictions, axis=1)

        self.accuracy = accuracy_score(t, p)
        self.loss = log_loss(self.labels, predictions)
        self.confusion_matrix = confusion_matrix(t, p)
        self.precision = precision_score(t, p)
        self.recall = recall_score(t, p)
        self.f1 = f1_score(t, p)

        logs['val_acc'] = self.accuracy
        logs['val_loss'] = self.loss
        logs['val_precision'] = self.precision
        logs['val_recall'] = self.recall
        logs['val_f1'] = self.f1

        print('\nEvaluating for epoch {}...'.format(epoch + 1))
        print('Confusion Matrix:\n', self.confusion_matrix)
        print('Accuracy: {:.4f}'.format(self.accuracy))
        print('Loss: {:.4f}'.format(self.loss))
        print('Precision: {:.4f}'.format(self.precision))
        print('Recall: {:.4f}'.format(self.recall))
        print('F-score: {:.4f}'.format(self.f1))
