'''
TODO: Build bidrectional LSTM to classify hate speech
Backend: Tensorflow
Dataset: https://www.crowdflower.com/data-for-everyone/
Contributors viewed short text and identified if it a) contained hate speech, b) was offensive but without hate speech, or c) was not offensive at all. Contains nearly 15K rows with three contributor judgments per text string.
'''

from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Dropout
import numpy as np

np.random_seed()

# import Dataset

print 'Loading data...'