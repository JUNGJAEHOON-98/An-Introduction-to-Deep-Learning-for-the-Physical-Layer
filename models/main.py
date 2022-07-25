# importing libs
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from autoencoder import AutoEncoder
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
# from keras.optimizers import SGD

def set_seed(seed: int = 25):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(1024)

# defining system parameters
M = 16
k = int(np.log2(M))
n_channel = 7
rate = k/n_channel
EbNo_train = 5.01187 #  coverted 7 db of EbNo

print('-----------SYSTEM PARAMETER-------------')
print ('M:', M)
print ('k', k)
print('Rate:', rate)
print('Number of channel use:', n_channel)
print('Eb_No[dB]',)
print('---------------------------------------\n')

# defining training parameters 
epochs = 20
batch_size = 256
lr = 0.002
N_train = 70000
N_val = 20000

optimizer = tf.keras.optimizers.Adam(learning_rate= lr) # optimizer = SGD()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

print('----------TRAINING PARAMETER------------')
print ('epochs:', epochs)
print ('batch_size', batch_size)
print('Optimizer:', optimizer)
print('Loss function:',loss_fn )
print('---------------------------------------\n\n')

#generating data of size N

train_data = np.random.randint(M, size=N_train)
train_label = to_categorical(train_data, num_classes=16)

val_data = np.random.randint(M, size=N_val)
val_label = to_categorical(val_data, num_classes=16)


autoencoder = AutoEncoder(M, n_channel, rate, EbNo_train)

train_loss_metrics = tf.keras.metrics.Mean()
train_acc_metrics = tf.keras.metrics.CategoricalAccuracy()

val_loss_metrics = tf.keras.metrics.Mean()
val_acc_metrics = tf.keras.metrics.CategoricalAccuracy()


def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = autoencoder(x)
        loss_value = loss_fn(y, predictions)
    gradients = tape.gradient(loss_value, autoencoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_weights))
 
    train_loss_metrics(loss_value)
    train_acc_metrics(y, predictions)

def valid_step(test_data, test_label):   
    predictions = autoencoder(test_data)
    loss_value = loss_fn(test_label, predictions)
    val_loss_metrics(loss_value)
    val_acc_metrics(test_label, predictions)

for epoch in range(epochs):

    for i in range(N_train // batch_size):
        x_batch = train_data[i*batch_size:(i+1)*batch_size]
        y_batch = train_label[i*batch_size:(i+1)*batch_size]
        train_step(x_batch, y_batch)

    valid_step(val_data, val_label)
    template = 'epoch: {}, train_loss: {:.5f}, train_acc: {:.2f}%, val_loss: {:.5f}, val_acc: {:.2f}%'
    
    print (template.format(epoch+1,
                        train_loss_metrics.result(),
                        train_acc_metrics.result()*100,
                        val_loss_metrics.result(),
                        val_acc_metrics.result()*100))



if n_channel == 2:
    scatter_plot = []
    for i in range(0,M):
        temp = np.zeros(M)
        temp[i] = 1
        output = autoencoder.encoder.predict(np.expand_dims(temp,axis=0))
        encoded_signal  = np.sqrt(n_channel) * tf.math.l2_normalize(output)

        scatter_plot.append(encoded_signal)
        
    # ploting constellation diagram
    scatter_plot = np.array(scatter_plot).reshape(-1, 2)
    plt.scatter(scatter_plot[:,0],scatter_plot[:,1])
    plt.axis((-2,2,-2,2))
    plt.grid()
    plt.xlabel('Q Axis')
    plt.ylabel('I Axis')
    plt.show()
    plt.savefig('AutoEncoder(2,2)_Constellations.png')


else:
    N = 20000
    test_label = np.random.randint(M,size=N)
    test_data = []

    for i in test_label:
        temp = np.zeros(M)
        temp[i] = 1
        test_data.append(temp)
        
    test_data = np.array(test_data)

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    EbNodB_range = list(frange(-4,8.5,0.5))
    ber = [None]*len(EbNodB_range)
    for n in range(0,len(EbNodB_range)):
        errors_sum = 0

        EbNo=10.0**(EbNodB_range[n]/10.0)
        noise_std = np.sqrt(1/(2*rate*EbNo))

        encoded_signal = autoencoder.encoder.predict(test_data) 
        encoded_signal  = np.sqrt(n_channel)*tf.math.l2_normalize(encoded_signal, axis=-1)

        noise = noise_std * np.random.randn(encoded_signal.shape[0], n_channel)
        channel_output = encoded_signal + noise

        pred_final_signal =  autoencoder.decoder.predict(channel_output)
        pred_output = np.argmax(pred_final_signal,axis=1)

        errors = (pred_output != test_label)
        errors_sum =  errors.astype(int).sum()

        ber[n] = errors_sum / N
        print ('SNR:',EbNodB_range[n],'BER:',ber[n])

    plt.plot(EbNodB_range, ber, 'bo',label='Autoencoder(7,4)')
    #plt.plot(list(EbNodB_range), ber_theory, 'ro-',label='BPSK BER')
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc='upper right',ncol = 1)
    plt.savefig('BLER versus Eb/No for (7,4) AutoEncoder.png')