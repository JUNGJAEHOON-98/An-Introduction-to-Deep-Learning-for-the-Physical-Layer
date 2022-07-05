from keras.layers import Dense

class Receiver():
    """
    Converts the channel_output vector to decoded vector 

    Inputs:
        - input_signal: channel_output signal
        - input_dim: M-dimension (M-dimensional input_signal)
    
    Returns: output
        - output: estimate symbol (decoded symbol)
                  probability distribution over all possible messages from which the most likely is picked as output
    """

    def __init__(self, input_signal, input_dim) -> None:
        super(Receiver, self).__init__()
        self.input_signal = input_signal
        self.input_dim = input_dim
        self.dense1 = Dense(self.input_dim, activation='relu')
        self.dense2 = Dense(self.input_dim, activation='softmax')

    def forward(self):
        output = self.dense1(self.input_signal)
        output = self.dense2(output)

        return output  