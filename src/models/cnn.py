from typing import Optional, Tuple

import numpy as np

from keras import Model
from keras.layers import Embedding, Input, Conv1D, Dropout, Concatenate, TimeDistributed, Dense


class CNNModel(Model):
    def __init__(self,
                 nb_symbols: Optional[int]=None,
                 embeddings_size: Optional[int]=None,
                 window_size: Tuple[int]=(5, 5, 5),
                 filters_number: Tuple[int]=(192, 192, 192),
                 dense_output_units: int=0,
                 dropout: float=0.,
                 nb_classes: int=15,
                 inputs=None, outputs=None, name='SegmentationNetwork'):
        if inputs or outputs:
            super(CNNModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        filters_number = np.atleast_2d(filters_number)

        net_input = Input(shape=(None,), dtype='uint8', name="input")
        char_embeddings = Embedding(nb_symbols, embeddings_size, name="char_embeddings")(net_input)
        conv_inputs = char_embeddings
        conv_outputs = []
        for window_size, curr_filters_numbers in zip(window_size, filters_number):
            curr_conv_input = conv_inputs
            for j, filters_number in enumerate(curr_filters_numbers[:-1]):
                curr_conv_input = Conv1D(filters_number, window_size, activation="relu", padding="same")(curr_conv_input)
                if dropout > 0.0:
                    curr_conv_input = Dropout(dropout)(curr_conv_input)
            curr_conv_output = Conv1D(curr_filters_numbers[-1], window_size, activation="relu", padding="same")(curr_conv_input)
            conv_outputs.append(curr_conv_output)

        if len(conv_outputs) == 1:
            conv_output = conv_outputs[0]
        else:
            conv_output = Concatenate(name="conv_output")(conv_outputs)
        if dense_output_units:
            pre_last_output = TimeDistributed(
                Dense(dense_output_units, activation="relu"),
                name="pre_output")(conv_output)
        else:
            pre_last_output = conv_output

        output = TimeDistributed(Dense(nb_classes, activation="softmax"), name="output")(pre_last_output)
        super(CNNModel, self).__init__(inputs=[net_input], outputs=[output], name=name)
