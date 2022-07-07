# importing libs
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers, Sequential
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.models import Model
import random as rn


# defining parameters
M = 16 
k = np.log2(M)
k = int(k)
print ('M:',M,'k:',k)

#generating data of size N
N = 20000
train_data = np.random.randint(M,size=N)
train_label = to_categorical(train_data)
#train_data = to_categorical(label) # convert one-hot vector (ndarray)

N_val = 5000
val_data = np.random.randint(M,size=N_val)
val_label = to_categorical(val_data)

rate = 4/7
n_channel = 7
EbNo_train = 5.01187 #  coverted 7 db of EbNo
print (int(k/rate))

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
        one_hat_vector = to_categorical(inputs)
        transmitted_signal = self.encoder(one_hat_vector)
        channel_output = self.channel(transmitted_signal)
        output = self.decoder(channel_output)

        return output
        


autoencoder = AutoEncoder(M, n_channel, rate, EbNo_train)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()


train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.CategoricalAccuracy()

test_loss = tf.keras.metrics.Mean()
test_acc = tf.keras.metrics.CategoricalAccuracy()

# def train_step(data, label):
#     for epoch_index in range(epochs):
#         with tf.GradientTape() as tape:

#             predictions = autoencoder(data)
#             loss_value = loss(label, predictions)

#         gradients = tape.gradient(loss_value, autoencoder.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
#         print('epoch: {}/{}: loss: {:.4f}'.format(epoch_index + 1, epochs, loss_value.numpy()))

def train_step(data, label):

    with tf.GradientTape() as tape:

        predictions = autoencoder(data)
        loss_value = loss(label, predictions)

    gradients = tape.gradient(loss_value, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    train_loss(loss_value)
    train_acc(label, predictions)



def test_step(data, label):

    predictions = autoencoder(data)
    loss_value = loss(label, predictions)
    test_loss(loss_value)
    test_acc(label, predictions)

epochs = 1000 
    
for epoch in range(epochs):

    train_step(train_data, train_label)
    test_step(val_data, val_label)
    template = 'epoch: {}, train_loss: {:.5f}, train_acc: {:.2f}%, test_loss: {:.5f}, test_acc: {:.2f}%'
    if epoch % 9 == 0: 
        print (template.format(epoch+1,
                            train_loss.result(),
                            train_acc.result()*100,
                            test_loss.result(),
                            test_acc.result()*100))