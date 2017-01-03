'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
Reviews have been preprocessed, and each review is encoded as a sequence of word 
indexes (integers). For convenience, words are indexed by overall frequency in the 
dataset, so that for instance the integer "3" encodes the 3rd most frequent word

As a convention, "0" does not stand for a specific word, but instead is used to 
encode any unknown word.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducible data shuffling

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.datasets import imdb


max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features) #nb means number
# nb_words=10000 Only consider the top 10,000 most common words
# nb_words = None Considers all words
# skip_top=20 Eliminate the top 20 most common words 
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape) # these are the sequences which are lists of indexes that represent words and their frequency
print('X_test shape:', X_test.shape) # same
y_train = np.array(y_train) # these are the integer labels (0 or 1)
y_test = np.array(y_test) # these are the integer labels (0 or 1)

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
