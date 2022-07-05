import tensorflow as tf
from channel import Channel
from transmitter import Transmitter
from receiver import Receiver


class AutoEncoder():
    def __init__(self, input_signal, input_dim, num_channels, rate, Eb_N0) -> None:
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_channel = num_channels
        self.input_signal = input_signal
        self.rate = rate 
        self.Eb_N0 = Eb_N0
        self.channel = Channel()
        self.transmitter = Transmitter()
        self.receiver = Receiver()
        
    def forward(self):
        transmitted_signal = Transmitter(self.input_signal, self.input_dim, self.num_channel)
        ch_output = Channel(transmitted_signal, self.rate, self.Eb_N0)
        output = Receiver(ch_output, self.input_dim)

        return output
