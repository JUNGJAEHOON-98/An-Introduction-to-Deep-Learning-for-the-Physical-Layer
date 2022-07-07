import numpy as np
import tensorflow as tf 
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical

class Transmitter(Model):
    """
    Converts the transmitted symbols to n-dimensional normalized transmitted vector (n: # of channel)

    Inputs:
        - input_signal: transmitted symbols(vector s)
        - input_size (int): dimension of transmitted symbols 
        - num_channels (int): the number of channel 

    Returns: output
        - output: n-dimensional normalized vector
    """

    def __init__(self, M, num_channels):
        super(Transmitter, self).__init__()
        #self.embedding = to_categorical()
        self.dense1 = Dense(M, activation='relu')
        self.dense2 = Dense(num_channels, activation='linear')
        self.layernorm = BatchNormalization()
        
    def forward(self, inputs):
        print('1111111111111')
        one_hot_vector = to_categorical(inputs)  # convert symbol to M-dimensional one-hot vector (ndarray)
        print('shape_one-hot-vector:', one_hot_vector.shape)
        output = self.dense1(one_hot_vector)
        output = self.dense2(output)

        output = self.layernorm(output)

        return output

