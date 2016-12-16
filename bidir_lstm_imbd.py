'''Trains a bidirectional LSTM on IMDB dataset for sentiment classification.

50,000 movies reviews labeled positive/negative -distribution is equal
Each review is encoded as a sequence of word indexes (integers)
words are indexed by overall frequency in the dataset, ie
"3" encodes the 3rd most frequent word in the data. 

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU: ~1500s.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337) # try with empty

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.datasets import imdb

max_features = 20000
maxlen = 100  # cut texts after this number of words (from the top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
# nb_words=10000 Only consider the top 10,000 most common words
# nb_words = None Considers all words
# skip_top=20 Eliminate the top 20 most common words 
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=4,
          validation_data=[X_test, y_test])



'''
Initial result:
Using TensorFlow backend.
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb_full.pkl
65560576/65552540 [==============================] - 148s
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
X_train shape: (25000, 100)
X_test shape: (25000, 100)
Train...
Train on 25000 samples, validate on 25000 samples
Epoch 1/4
25000/25000 [==============================] - 1033s - loss: 0.4197 - acc: 0.8047 - val_loss: 0.3397 - val_acc: 0.8499
Epoch 2/4
25000/25000 [==============================] - 1187s - loss: 0.2341 - acc: 0.9085 - val_loss: 0.3427 - val_acc: 0.8485
Epoch 3/4
25000/25000 [==============================] - 1059s - loss: 0.1307 - acc: 0.9533 - val_loss: 0.4438 - val_acc: 0.8337
Epoch 4/4
25000/25000 [==============================] - 1065s - loss: 0.0672 - acc: 0.9786 - val_loss: 0.5635 - val_acc: 0.8363


Validation data is the test data--looks like this model uses the same data for training :(
If training data accuracy (acc) keeps improving while validation data accuracy (val_acc) 
gets worse, you are likely in an overfitting situation,

If your training loss is much lower than validation loss then this means the network might be 
overfitting. Solutions to this are to decrease your network size (# hidden layers and their sizes), 
or to increase dropout. For example you could try dropout of 0.5 and so on.

If your training/validation loss are about equal then your model is underfitting. Increase the size of your model 
(either number of layers or the raw number of neurons per layer)

Each epoch is like it's own model. You can use the Keras callback ModelCheckpoint to automatically save the model with 
the highest validation accuracy

The lower the Loss, the better a model (unless the model has over-fitted to the training data). The loss is calculated 
on training and validation and its interperation is how well the model is doing for these two sets. Loss is not in 
percentage as opposed to accuracy and it is a summation of the errors made for each example in training or validation 
sets. In the case of neural networks the loss is usually negative log-likelihood and residual sum of squares for 
classification and regression respectively. 

Dense is normal fully connected nn layer
Dropout randomly sets a fraction p of input units to 0 at each update during 
training to prevent overfitting.
Embedding turns positive integers (indexes) into dense vectors of fixed size. 
eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
Long-Short Term Memory unit is a recurrent layer (more below)
Bidirectional-this is a wrapper for LSTM and has merge modes for how to combine the 
outputs of the forward and backward RNN. One of {'sum', 'mul', 'concat', 'ave', None}.


LSTM model has a memory cell composed of four main elements: 
an input gate, a neuron with a self-recurrent connection (a connection to 
itself), a forget gate and an output gate. The self-recurrent connection has a weight of 1.0 
and ensures that, barring any outside interference, the state of a memory cell can remain 
constant from one timestep to another. The gates serve to modulate the interactions between 
the memory cell itself and its environment. The input gate can allow incoming signal to alter 
the state of the memory cell or block it. On the other hand, the output gate can allow the 
state of the memory cell to have an effect on other neurons or prevent it. Finally, the forget 
gate can modulate the memory cell’s self-recurrent connection, allowing the cell to remember 
or forget its previous state, as needed.
'''


