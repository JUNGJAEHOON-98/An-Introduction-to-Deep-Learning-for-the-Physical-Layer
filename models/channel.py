from turtle import forward
from keras.models import Model
import numpy as np
from keras.layers import GaussianNoise

class Channel(Model):
    """
    Additive White Gaussian Noise(AWGN) Channel

    Inputs:
        - input_signal: transmitted signal(vector x)
        - rate: data rate
        - Eb-N0: energy per bit to noise power spectral density ratio

    Returns: output
        - channel_output: n-dimensional channel output vector
    """

    def __init__(self, rate, Eb_N0) -> None:
        super(Channel, self).__init__()
        self.rate = rate
        self.Eb_N0 = Eb_N0
        self.variance = np.sqrt(1/(2*self.rate*self.Eb_N0))
        self.channel = GaussianNoise(self.variance)

    def forward(self, inputs):
        channel_output = self.channel(inputs)

        return channel_output

