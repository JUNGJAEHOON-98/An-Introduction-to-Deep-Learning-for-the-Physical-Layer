from turtle import forward
import numpy as np
from keras.layers import GaussianNoise

class Channel():
    """
    Additive White Gaussian Noise(AWGN) Channel

    Inputs:
        - input_signal: transmitted signal(vector x)
        - rate: data rate
        - Eb-N0: energy per bit to noise power spectral density ratio

    Returns: output
        - channel_output: n-dimensional channel output vector
    """

    def __init__(self, rate, input_signal, Eb_N0) -> None:
        self.rate = rate
        self.Eb_N0 = Eb_N0
        self.input_signal = input_signal
        self.variance = np.sqrt(1/(2*self.rate*self.Eb_N0))
        self.channel = GaussianNoise(self.variance)

    def forward(self):
        channel_output = self.channel(self.input_signal)

        return channel_output

