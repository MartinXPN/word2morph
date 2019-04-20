from typing import Tuple

from keras import Model
from keras.layers import Embedding, Input, Conv1D, Dropout, TimeDistributed, Dense
from keras_contrib.layers import CRF


class CNNModel(Model):
    def __init__(self,
                 nb_symbols: int = 34,
                 embeddings_size: int = 8,
                 kernel_sizes: Tuple[int, ...] = (5, 5, 5),
                 nb_filters: Tuple[int, ...] = (192, 192, 192),
                 dilations: Tuple[int, ...] = (1, 1, 1),
                 dense_output_units: int = 0,
                 dropout: float = 0.,
                 use_crf: bool = True,
                 nb_classes: int = 15,
                 inputs=None, outputs=None, name='SegmentationNetwork'):
        if inputs or outputs:
            super(CNNModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        net_input = Input(shape=(None,), dtype='uint8', name='input')
        char_embeddings = Embedding(nb_symbols, embeddings_size, name='char_embeddings')(net_input)

        x = char_embeddings
        for kernel_size, filters, dilation in zip(kernel_sizes[:-1], nb_filters[:-1], dilations[:-1]):
            x = Conv1D(filters, kernel_size, activation='relu', padding='same', dilation_rate=dilation)(x)
            x = Dropout(dropout)(x)
        x = Conv1D(nb_filters[-1], kernel_sizes[-1], activation='relu', padding='same', dilation_rate=dilations[-1])(x)

        x = TimeDistributed(Dense(dense_output_units, activation='relu'), name='pre_output')(x) \
            if dense_output_units else x

        output = CRF(units=nb_classes, learn_mode='join', name='output')(x) \
            if use_crf else TimeDistributed(Dense(nb_classes, activation='softmax', name='output'))(x)
        super(CNNModel, self).__init__(inputs=[net_input], outputs=[output], name=name)
