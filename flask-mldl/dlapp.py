from flask import Flask
from flask import request
import pandas as pd 
import numpy as np 
import pickle
from helperClasses import *
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from collections import Counter
import tensorflow as  tf
import keras
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def process_description(t, test_file=True):

	#Generates training X, y pairs.

	tweets = []
	feature_vector = get_userfeature_vector(t)
	tweets.append(feature_vector)
	return tweets

def process_tweet(t, test_file=True):

	#Generates training X, y pairs.

	tweets = []
	feature_vector = get_tweetfeature_vector(t)
	tweets.append(feature_vector)
	return tweets


def get_userfeature_vector(user):

   # Generates a feature vector for each tweet where each word is represented by integer index based on rank in vocabulary.

	words = user.split()
	feature_vector = []
	for i in range(len(words) - 1):
		word = words[i]
		if vocabu.get(word) is not None:
			feature_vector.append(vocabu.get(word))
	if len(words) >= 1:
		if vocabu.get(words[-1]) is not None:
			feature_vector.append(vocabu.get(words[-1]))
	return feature_vector

def get_tweetfeature_vector(tweet):

   # Generates a feature vector for each tweet where each word is represented by integer index based on rank in vocabulary.

	words = tweet.split()
	feature_vector = []
	for i in range(len(words) - 1):
		word = words[i]
		if vocabt.get(word) is not None:
			feature_vector.append(vocabt.get(word))
	if len(words) >= 1:
		if vocabt.get(words[-1]) is not None:
			feature_vector.append(vocabt.get(words[-1]))
	return feature_vector

def top_n_words(tweetm,n,shift=0):

	freq_dict = {}
	tweet=' '.join(tweetm)
	words = tweet.split()
	for word in words:
		if freq_dict.get(word):
			freq_dict[word] += 1
		else:
			freq_dict[word] = 1
	freq_dist = Counter(freq_dict)
	most_common = freq_dist.most_common(n)
	words = {p[0]: i + shift for i, p in enumerate(most_common)}
	return words

app = Flask(__name__)

vocab_size=90000
mDataFrame = pd.read_csv("test1.csv").fillna('')
userdescription=mDataFrame['user_description']
tweett=mDataFrame['tweet']
vocabu=top_n_words(userdescription,vocab_size,shift=1)
vocabt=top_n_words(tweett,vocab_size,shift=1)

@app.route('/cnnuser', methods=['GET','POST'])
def CNNuser():
	keras.backend.clear_session()
	max_length=40
	model = load_model("4cnnUser.hdf5")

	#modify it for the data stream
	text="hi how are you"
	test_tweets=process_description(text)
	test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
	predictions = model.predict(test_tweets, batch_size=128, verbose=1)
	results =  np.argmax(predictions, axis=1).astype(int)
	le = preprocessing.LabelEncoder()
	le.fit(["public","interest groups","media","businesses","celebrities","official agencies"])
	result=str(le.inverse_transform(results))
	return result

@app.route('/cnntweet', methods=['GET','POST'])
def CNNtweet():
	keras.backend.clear_session()
	model = load_model("4cnnTweet.hdf5")
	max_length = 140
	#modify it for the data stream
	text="hi how are you"
	test_tweets=process_tweet(text)
	test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
	predictions = model.predict(test_tweets, batch_size=128, verbose=1)
	results =  np.argmax(predictions, axis=1).astype(int)
	le = preprocessing.LabelEncoder()
	le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
	result=str(le.inverse_transform(results))
	return result

@app.route('/lstmuser', methods=['GET','POST'])
def LSTMuser():
	keras.backend.clear_session()
	max_length=40
	model = load_model("LstmUser.hdf5")

	#modify it for the data stream
	text="hi how are you"
	test_tweets=process_description(text)
	test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
	predictions = model.predict(test_tweets, batch_size=128, verbose=1)
	results =  np.argmax(predictions, axis=1).astype(int)
	le = preprocessing.LabelEncoder()
	le.fit(["public","interest groups","media","businesses","celebrities","official agencies"])
	result=str(le.inverse_transform(results))
	return result

@app.route('/lstmtweet', methods=['GET','POST'])
def lstmtweet():
	keras.backend.clear_session()
	max_length=140
	model = load_model("LstmTweet.hdf5")

	#modify it for the data stream
	text="hi how are you"
	test_tweets=process_tweet(text)
	test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
	predictions = model.predict(test_tweets, batch_size=128, verbose=1)
	results =  np.argmax(predictions, axis=1).astype(int)
	le = preprocessing.LabelEncoder()
	le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
	result=str(le.inverse_transform(results))
	return result


if __name__ == '__main__':
	app.run(debug=True, port=33507)