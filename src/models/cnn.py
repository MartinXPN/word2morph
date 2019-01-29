from typing import Optional, Tuple

from keras import Model
from keras.layers import Embedding, Input, Conv1D, Dropout, TimeDistributed, Dense


class CNNModel(Model):
    def __init__(self,
                 nb_symbols: Optional[int]=None,
                 embeddings_size: Optional[int]=None,
                 kernel_sizes: Tuple[int]=(5, 5, 5),
                 nb_filters: Tuple[int]=(192, 192, 192),
                 dense_output_units: int=0,
                 dropout: float=0.,
                 nb_classes: int=15,
                 inputs=None, outputs=None, name='SegmentationNetwork'):
        if inputs or outputs:
            super(CNNModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        net_input = Input(shape=(None,), dtype='uint8', name="input")
        char_embeddings = Embedding(nb_symbols, embeddings_size, name="char_embeddings")(net_input)

        x = char_embeddings
        for kernel_size, filters in zip(kernel_sizes[:-1], nb_filters[:-1]):
            x = Conv1D(filters, kernel_size, activation="relu", padding="same")(x)
            x = Dropout(dropout)(x)
        x = Conv1D(nb_filters[-1], kernel_sizes[-1], activation="relu", padding="same")(x)

        output = TimeDistributed(Dense(dense_output_units, activation="relu"), name="pre_output")(x) \
            if dense_output_units else x

        output = TimeDistributed(Dense(nb_classes, activation="softmax"), name="output")(output)
        super(CNNModel, self).__init__(inputs=[net_input], outputs=[output], name=name)
