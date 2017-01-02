'''
TODO: Build bi-drectional recurrent neural network (BRNN) to classify hate speech
Specifically, a bi-directional long short-term memory (LSTM) network
Backend: Tensorflow
Dataset: https://www.crowdflower.com/data-for-everyone/
Twitter
3 contributors viewed short text and identified if it a) contained hate speech, b) was offensive but without hate speech, 
or c) was not offensive at all. Contains nearly 15K rows with three contributor judgments per text string.
'''

from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Dropout
import numpy as np

np.random.seed()



# import Dataset 'twitter_hate.csv'

print ('Loading data...')
(x_train, y_train), (x_test, y_test) = #load_data(nb_words=20000 , skip_top= 20)
print (len(x_train), 'positive training sequences') 
#"The tweet is not offensive" (7320) or "The tweet uses offensive language but not hate speech" (4907)
print (len(y_train), 'negative training sequences') # "The tweet contains hate speech"
print (len(x_test), 'positive test sequences')
print (len(y_test), 'negative test sequences')



model = Sequential([
Embedding(1000, 64, input_length=10), #vocab size, batch, input_length
Bidirectional(LSTM(64),
Dropout(0.5))
Dense(1, activation='Sigmoid') # a layer where each unit or neuron is connected to each neuron in the next layer
]),

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)

print ('Training...')
model.fit(x_train, y_train, 
	batch_size=#, 
	nb_epoch =1, 
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

