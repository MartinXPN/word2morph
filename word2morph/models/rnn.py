from typing import Tuple

from keras import Model
from keras.layers import Embedding, Input, Dropout, TimeDistributed, Dense, GRU, Bidirectional, Masking
from keras_contrib.layers import CRF


class RNNModel(Model):
    def __init__(self,
                 nb_symbols: int = 34,
                 embeddings_size: int = 8,
                 recurrent_units: Tuple[int, ...] = (64, 128, 256),
                 dense_output_units: int = 0,
                 dropout: float = 0.,
                 nb_classes: int = 15,
                 use_crf: bool = True,
                 inputs=None, outputs=None, name='SegmentationNetwork'):
        if inputs or outputs:
            super(RNNModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        net_input = Input(shape=(None,), dtype='uint8', name='input')
        char_embeddings = Embedding(input_dim=nb_symbols,
                                    output_dim=embeddings_size,
                                    name='char_embeddings',
                                    mask_zero=True)(net_input)

        x = Masking()(char_embeddings)
        for units in recurrent_units:
            x = Bidirectional(GRU(units=units, return_sequences=True))(x)
            x = Dropout(dropout)(x)

        x = TimeDistributed(Dense(dense_output_units, activation='relu'), name='pre_output')(x) \
            if dense_output_units else x

        output = CRF(units=nb_classes, learn_mode='join', name='output')(x) \
            if use_crf else TimeDistributed(Dense(nb_classes, activation='softmax', name='output'))(x)

        super(RNNModel, self).__init__(inputs=[net_input], outputs=[output], name=name)
