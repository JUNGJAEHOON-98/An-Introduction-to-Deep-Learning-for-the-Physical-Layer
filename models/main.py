# importing libs
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from autoencoder import AutoEncoder
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from tqdm import tqdm
# from keras.optimizers import SGD


# defining system parameters
M = 16 
k = int(np.log2(M))
rate = 4/7
n_channel = 7
EbNo_train = 5.01187 #  coverted 7 db of EbNo

print('-----------SYSTEM PARAMETER-------------')
print ('M:', M)
print ('k', k)
print('Rate:', rate)
print('Number of channel use:', n_channel)
print('Eb_No[dB]',)
print('---------------------------------------\n')

# defining training parameters 
epochs = 1
batch_size = 20
N_train = 10000
N_val = 5000

optimizer = tf.keras.optimizers.Adam() # optimizer = SGD()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

print('----------TRAINING PARAMETER------------')
print ('epochs:', epochs)
print ('batch_size', batch_size)
print('Optimizer:', 'Adam')
print('Loss function:', 'CategoricalCrossentropy')
print('---------------------------------------\n\n')

#generating data of size N

train_data = np.random.randint(M, size=N_train)
train_label = to_categorical(train_data)
train_data = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(batch_size)

val_data = np.random.randint(M, size=N_val)
val_label = to_categorical(val_data)


autoencoder = AutoEncoder(M, n_channel, rate, EbNo_train)

train_loss_metrics = tf.keras.metrics.Mean()
train_acc_metrics = tf.keras.metrics.CategoricalAccuracy()

val_loss_metrics = tf.keras.metrics.Mean()
val_acc_metrics = tf.keras.metrics.CategoricalAccuracy()


def apply_gradient(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        predictions = autoencoder(x)
        loss_value = loss_fn(y, predictions)
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
 
    return predictions, loss_value


def train_for_one_epoch(train_data, model, optimizer, loss_fn, train_acc_metrics):

    one_epoch_loss = []
    one_epoch_acc = []

    for epoch, (x_batch, y_batch) in enumerate(train_data):
        predictions, loss_value = apply_gradient(model, optimizer, loss_fn, x_batch, y_batch)       
       
 
        one_epoch_loss.append(train_loss_metrics(loss_value))
        one_epoch_acc.append(train_acc_metrics(y_batch, predictions))

    return np.mean(one_epoch_loss), np.mean(one_epoch_acc)

    

def perform_validation(test_data, test_label, model, loss_fn, valid_acc_metrics):   
    predictions = model(test_data)

    loss_value = loss_fn(test_label, predictions)
    acc_value = valid_acc_metrics(test_label, predictions)

    return loss_value, acc_value
    

val_losses, train_losses = [], []

for epoch in range(epochs):
    print(f'Start of epoch {epoch+1}')
  
    train_loss, train_acc = train_for_one_epoch(train_data, autoencoder, optimizer, loss_fn, train_acc_metrics)
    val_loss, val_acc = perform_validation(val_data, val_label, autoencoder, loss_fn, val_acc_metrics)
  
    template = 'epoch: {}, train_loss: {:.5f}, train_acc: {:.2f}%, val_loss: {:.5f}, val_acc: {:.2f}%'

    print (template.format(epoch+1,
                        train_loss,
                        train_acc*100,
                        val_loss,
                        val_acc*100))

    train_losses.append(train_loss_metrics.result())
    val_losses.append(val_loss_metrics.result())


def plot_metrics(train_metric, valid_metric, title):
    plt.title(title)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.plot(train_metric, color='red', label="train_loss")
    plt.plot(valid_metric, color='blue', label= "val_loss")
    plt.legend()
 
plot_metrics(train_losses, val_losses, "Loss Value")

N = 45000
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
    EbNo=10.0**(EbNodB_range[n]/10.0)
    noise_std = np.sqrt(1/(2*rate*EbNo))
    noise_mean = 0
    no_errors = 0
    nn = N
    noise = noise_std * np.random.randn(nn,n_channel)
    encoded_signal = autoencoder.encoder.predict(test_data) 
    final_signal = encoded_signal + noise
    pred_final_signal =  autoencoder.decoder.predict(final_signal)
    pred_output = np.argmax(pred_final_signal,axis=1)
    no_errors = (pred_output != test_label)
    no_errors =  no_errors.astype(int).sum()
    ber[n] = no_errors / nn 
    print ('SNR:',EbNodB_range[n],'BER:',ber[n])

plt.plot(EbNodB_range, ber, 'bo',label='Autoencoder(7,4)')
#plt.plot(list(EbNodB_range), ber_theory, 'ro-',label='BPSK BER')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right',ncol = 1)
plt.savefig('AutoEncoder_7_4_BER_matplotlib')