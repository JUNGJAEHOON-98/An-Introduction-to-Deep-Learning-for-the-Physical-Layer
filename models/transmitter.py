import numpy as np
import tensorflow as tf 
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.utils import to_categorical

class Transmitter():
    """
    Converts the transmitted symbols to n-dimensional normalized transmitted vector (n: # of channel)

    Inputs:
        - input_signal: transmitted symbols(vector s)
        - input_size (int): dimension of transmitted symbols 
        - num_channels (int): the number of channel 

    Returns: output
        - output: n-dimensional normalized vector
    """

    def __init__(self, input_signal, input_dim, num_channels):
        super(Transmitter, self).__init__()
        self.input_signal = input_signal
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.dense1 = Dense(self.input_dim, activation='relu')
        self.dense2 = Dense(self.num_channels, activation='linear')
        self.layernorm = BatchNormalization()
        
    def forward(self):
        one_hot_vector = to_categorical(self.input_signal)

        output = self.dense1(one_hot_vector)
        output = self.dense2(output)

        output = self.layernorm(output)

        return output

