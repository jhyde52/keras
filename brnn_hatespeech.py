'''
TODO: Build bi-drectional recurrent neural network (BRNN) to classify hate speech
Specifically, a bi-directional long short-term memory (LSTM) network
Backend: Tensorflow
Dataset: https://www.crowdflower.com/data-for-everyone/
Twitter
3 contributors viewed short text and identified if it a) contained hate speech, b) was offensive but without hate speech, 
or c) was not offensive at all. Contains nearly 15K rows with three contributor judgments per text string.

Steps:
1. Preprocess data: Turn words into indexes (preprocessing.one_hot?)
https://docs.google.com/spreadsheets/d/1kyAwzzLIjdXXc5Q6R0GLUqXkIH5Yvyuk8zHOZy7Ttng/edit#gid=0

2. Add a series of layers and parameters

3. Train, adjust parameters, train


'''

from __future__ import print_function
import numpy as np
np.random.seed(1432)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Dropout

# import Dataset 'twitter_hate.csv' -- maybe write another script to preprocess it first
# Encode as 0 for negative - "The tweet is not offensive" (7320) or "The tweet uses offensive language but not hate speech" (4907)
# Encode as 1 for positive - "The tweet contains hate speech"


print ('Loading data...')
(x_train, y_train), (x_test, y_test) = #load_data(nb_words=TDB , skip_top=TDB, etc)
# nb_words=10000 Only consider the top 10,000 most common words
# nb_words = None Considers all words
# skip_top=20 Eliminate the top 20 most common words

print (len(x_train), 'positive training sequences') 
print (len(y_train), 'negative training sequences') 
print (len(x_test), 'positive test sequences')
print (len(y_test), 'negative test sequences')

'''! Need to update why capital x?  !
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape) # these are the sequences which are lists of indexes that represent words and their frequency
print('X_test shape:', X_test.shape)
y_train = np.array(y_train) # these are the integer labels (0 or 1)
y_test = np.array(y_test)
'''



model = Sequential([
Embedding(1000, 64, input_length=10), #vocab size, batch, input_length
Bidirectional(LSTM(64),
Dropout(0.5)) # sets a fraction p of input units to 0 at each update during training to prevent overfitting.
Dense(1, activation='sigmoid') # a layer where each unit or neuron is connected to each neuron in the next layer
]),

input_array = np.random.randint(1000, size=(32, 10))


# try using different optimizers and different optimizer configs like ADAM and binary_crossentropy
model.compile('rmsprop', 'mse', metrics = 'accuracy')

# ?
# output_array = model.predict(input_array)
# assert output_array.shape == (32, 10, 64)

print ('Training...')
model.fit(x_train, y_train, 
	batch_size=#TBD, 
	nb_epoch =3, 
	validation_data=[x_test,y_test])



'''
_unit_id,
_golden,
_unit_state,
_trusted_judgments,
_last_judgment_at,
does_this_tweet_contain_hate_speech,
does_this_tweet_contain_hate_speech:confidence,
_created_at,orig__golden,orig__last_judgment_at,
orig__trusted_judgments,orig__unit_id,orig__unit_state,_updated_at,
orig_does_this_tweet_contain_hate_speech,does_this_tweet_contain_hate_speech_gold,
does_this_tweet_contain_hate_speech_gold_reason,
does_this_tweet_contain_hate_speechconfidence,
tweet_id,
tweet_text

The tweet uses offensive language but not hate speech",,1,203816023,"@AndreBerto word is you use roids, stupid hypocrite lying faggot."
853718229,TRUE,golden,94,,The tweet contains hate speech,0.8435,,TRUE,,0,615563683,golden,
,The tweet contains hate speech,The tweet contains hate speech,,1,395623778,I hate faggots like you

'''

