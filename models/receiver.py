from keras.layers import Dense
from keras.models import Model

class Receiver(Model):
    """
    Converts the channel_output vector to decoded vector 

    Inputs:
        - input_signal: channel_output signal
        - input_dim: M-dimension (M-dimensional input_signal)
    
    Returns: output
        - output: estimate symbol (decoded symbol)
                  probability distribution over all possible messages from which the most likely is picked as output
    """

    def __init__(self, input_dim) -> None:
        super(Receiver, self).__init__()
        self.dense1 = Dense(input_dim, activation='relu')
        self.dense2 = Dense(input_dim, activation='softmax')

    def forward(self, inputs):
        output = self.dense1(inputs)
        output = self.dense2(output)

        return output  