import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, GaussianNoise, BatchNormalization
from keras import Sequential
from tensorflow.keras.utils import to_categorical

class AutoEncoder(Model):
    def __init__(self, M, num_channels,  rate, Eb_N0):
        super(AutoEncoder, self).__init__()
        self.variance = np.sqrt(1/(2*rate*Eb_N0))

        self.encoder = Sequential([
            Dense(M, activation='relu', name="encoder_layer1"),
            Dense(num_channels, activation='linear', name="encoder_layer2"),
            BatchNormalization()  
        ])

        self.channel = GaussianNoise(self.variance)

        self.decoder = Sequential([
            Dense(M, activation='relu', name="decoder_layer1"),
            Dense(M, activation='softmax', name="decoder_layer2")
        ])

    def call(self, inputs):
        one_hat_vector = to_categorical(inputs, num_classes=16)
        transmitted_signal = self.encoder(one_hat_vector)
        channel_output = self.channel(transmitted_signal)
        output = self.decoder(channel_output)

        return output