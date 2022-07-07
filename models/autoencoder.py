import tensorflow as tf
from channel import Channel
from transmitter import Transmitter
from receiver import Receiver
from keras.models import Model


class AutoEncoder(Model):
    def __init__(self, input_dim, num_channels, rate, Eb_N0) -> None:
        super(AutoEncoder, self).__init__()
        self.channel = Channel(rate, Eb_N0)
        self.transmitter = Transmitter(input_dim, num_channels)
        self.receiver = Receiver(input_dim)
        
    def forward(self, inputs):
        transmitted_signal = Transmitter(inputs)
        ch_output = Channel(transmitted_signal)
        output = Receiver(ch_output)

        return output
