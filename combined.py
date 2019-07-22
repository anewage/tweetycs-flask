from flask import Flask
from flask import request,jsonify
import pandas as pd 
import numpy as np 
import pickle
from helperClasses import *
import json
from pandas.io.json import json_normalize
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

def load_lstmUsermodel():
    #keras.backend.clear_session()
    global clfLU
    clfLU = load_model("models/LstmUser.hdf5")
    global graph1
    graph1 = tf.get_default_graph()

def load_lstmTweetmodel():
    #keras.backend.clear_session()
    global clfLT
    clfLT =load_model("models/LstmTweet.hdf5")
    global graph2
    graph2 = tf.get_default_graph()

def load_cnnUsermodel():
    #keras.backend.clear_session()
    global clfCU
    clfCU = load_model("models/4cnnUser.hdf5")
    global graph3
    graph3 = tf.get_default_graph()

def load_cnnTweetmodel():
    #keras.backend.clear_session()
    global clfCT
    clfCT =load_model("models/4cnnTweet.hdf5")
    global graph4
    graph4 = tf.get_default_graph()



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
    if not user:
        words=[]
    else:
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
    if not tweet:
        words=[]
    else:
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
mDataFrame = pd.read_csv("data/test1.csv").fillna('')
userdescription=mDataFrame['user_description']
tweett=mDataFrame['tweet']
vocabu=top_n_words(userdescription,vocab_size,shift=1)
vocabt=top_n_words(tweett,vocab_size,shift=1)

def load_svmTweetmodel():
	global clfST
	model = open("models/SVMtweetOnly.pickle","rb")
	clfST = pickle.load(model)

def load_svmUsermodel():
	global clfSU
	model2 = open("models/SVMUserOnly.pickle","rb")
	clfSU=pickle.load(model2)

def load_knnTweetmodel():
	global clfKT
	model = open("models/KNNtweetOnly.pickle","rb")
	clfKT = pickle.load(model)

def load_knnUsermodel():
	global clfKU
	model2 = open("models/KNNUserOnly.pickle","rb")
	clfKU=pickle.load(model2)

def load_rfTweetmodel():
	global clfRT
	model = open("models/RFtweetOnly.pickle","rb")
	clfRT= pickle.load(model)

def load_rfUsermodel():
	global clfRU
	model2 = open("models/RFUserOnly.pickle","rb")
	clfRU=pickle.load(model2)

def load_mlpTweetmodel():
	global clfMT
	model = open("models/MLPtweetOnly.pickle","rb")
	clfMT= pickle.load(model)

def load_mlpUsermodel():
	global clfMU
	model2 = open("models/MLPUserOnly.pickle","rb")
	clfMU=pickle.load(model2)


@app.route('/svmtweet', methods=['GET','POST'])
def SVMtweet():
	if request.method =='POST':
		#model = open("SVMTweetOnly.pickle","rb")
		#clf = pickle.load(model)
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
		parameterIn=['tweet']
		X_test = df[parameterIn]
		bTest = clfST.predict(X_test)
		result=bTest
		result=result[0]
		return jsonify(theme=result)

@app.route('/svmuser', methods=['GET','POST'])
def SVMuser():
	if request.method =='POST':
		#model = open("SVMUserOnly.pickle","rb")
		#clf = pickle.load(model)
		#modify it for the data stream
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
		parameterIn=['user_description']
		test = df[parameterIn]
		bTest=clfSU.predict(test)
		result=bTest
		result=result[0]
		return jsonify(group=result)

@app.route('/knnuser', methods=['GET','POST'])
def KNNuser():
	if request.method =='POST':
		#model = open("KNNUserOnly.pickle","rb")
		#clf = pickle.load(model)
		#modify it for the data stream
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
		parameterIn=['user_description']
		test = df[parameterIn]
		bTest=clfKU.predict(test)
		result=bTest
		result=result[0]
		return jsonify(group=result)

@app.route('/knntweet', methods=['GET','POST'])
def KNNtweet():
	if request.method =='POST':
		# model = open("KNNTweetOnly.pickle","rb")
		# clf = pickle.load(model)
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
		#data=pd.read_json(data)
		#json_normalize(data['text'])
		parameterIn=['tweet']
		X_test = df[parameterIn]
		bTest = clfKT.predict(X_test)
		result=bTest
		result=result[0]
		return jsonify(theme=result)

@app.route('/rfuser', methods=['GET','POST'])
def RFuser():
	if request.method =='POST':
	#model = open("RFUserOnly.pickle","rb")
	#clf = pickle.load(model)
	#modify it for the data stream
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
		parameterIn=['user_description']
		test = df[parameterIn]
		bTest=clfRU.predict(test)
		result=bTest
		result=result[0]
		return jsonify(group=result)

@app.route('/rftweet', methods=['GET','POST'])
def RFtweet():
	if request.method =='POST':
	#model = open("RFTweetOnly.pickle","rb")
	#clf = pickle.load(model)
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
		parameterIn=['tweet']
		X_test = df[parameterIn]
		bTest = clfRT.predict(X_test)
		result=bTest
		result=result[0]
		return jsonify(theme=result)

@app.route('/mlpuser', methods=['GET','POST'])
def MLPuser():
	if request.method =='POST':
	#model = open("MLPUserOnly.pickle","rb")
	#clf = pickle.load(model)
	#modify it for the data stream
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
		parameterIn=['user_description']
		test = df[parameterIn]
		bTest=clfMU.predict(test)
		result=bTest
		result=result[0]
		return jsonify(group=result)


@app.route('/mlptweet', methods=['GET','POST'])
def MLPtweet():
	if request.method =='POST':
	#model = open("MLPTweetOnly.pickle","rb")
	#clf = pickle.load(model)
		data=request.get_json(force=True)
		df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
		parameterIn=['tweet']
		X_test = df[parameterIn]
		bTest = clfMT.predict(X_test)
		result=bTest
		result=result[0]
		return jsonify(theme=result)
    
@app.route('/cnnuser', methods=['GET','POST'])
def CNNuser():
    if request.method =='POST':
    #keras.backend.clear_session()
        max_length=40
    #model = load_model("4cnnUser.hdf5")
        if request.get_json():
    #modify it for the data stream
            data=request.get_json(force=True)
            text=data['user_description']
            test_tweets=process_description(text)
            test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
            with graph3.as_default():
                predictions = clfCU.predict(test_tweets, batch_size=128, verbose=1)
            results =  np.argmax(predictions, axis=1).astype(int)
            le = preprocessing.LabelEncoder()
            le.fit(["public","interest groups","media","businesses","celebrities","official agencies"])
            result=le.inverse_transform(results)
            result=result[0]
            return jsonify(group=result)

@app.route('/cnntweet', methods=['GET','POST'])
def CNNtweet():
    if request.method =='POST':
    #keras.backend.clear_session()
        # model = load_model("4cnnTweet.hdf5")
        max_length = 140
    #modify it for the data stream
        if request.get_json():
            data=request.get_json(force=True)
            text=data['tweet']
            test_tweets=process_tweet(text)
            test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
            with graph4.as_default():
                predictions = clfCT.predict(test_tweets, batch_size=128, verbose=1)
            results =  np.argmax(predictions, axis=1).astype(int)
            le = preprocessing.LabelEncoder()
            le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
            result=le.inverse_transform(results)
            result=result[0]
            return jsonify(theme=result)

@app.route('/lstmuser', methods=['GET','POST'])
def LSTMuser():
    if request.method =='POST':
    #keras.backend.clear_session()
        max_length=40
    #model = load_model("LstmUser.hdf5")
        if request.get_json():
    #modify it for the data stream
            data=request.get_json(force=True)
            text=data['user_description']
            test_tweets=process_description(text)
            test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
    #predictions = model.predict(test_tweets, batch_size=128, verbose=1)
            with graph1.as_default():
                predictions = clfLU.predict(test_tweets, batch_size=128, verbose=1)
    #keras.backend.clear_session()
            results =  np.argmax(predictions, axis=1).astype(int)
            le = preprocessing.LabelEncoder()
            le.fit(["public","interest groups","media","businesses","celebrities","official agencies"])
            result=le.inverse_transform(results)
            result=result[0]
            return jsonify(group=result)

@app.route('/lstmtweet', methods=['GET','POST'])
def lstmtweet():
    if request.method =='POST':
    #keras.backend.clear_session()
        max_length=140
    #model = load_model("LstmTweet.hdf5")
        if request.get_json():
    #modify it for the data stream
            data=request.get_json(force=True)
            text=data['tweet']
            test_tweets=process_tweet(text)
            test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
        #predictions = model.make_predict_function(test_tweets, batch_size=128, verbose=1)
        #keras.backend.clear_session()
            with graph2.as_default():
                predictions = clfLT.predict(test_tweets, batch_size=128, verbose=1)
            results =  np.argmax(predictions, axis=1).astype(int)
            le = preprocessing.LabelEncoder()
            le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
            result=le.inverse_transform(results)
            result=result[0]
            return jsonify(theme=result)


if __name__ == '__main__':
	load_svmTweetmodel()
	load_svmUsermodel()
	load_knnTweetmodel()
	load_knnUsermodel()
	load_rfTweetmodel()
	load_rfUsermodel()
	load_mlpTweetmodel()
	load_mlpUsermodel()
  load_lstmUsermodel()
  load_lstmTweetmodel()
  load_cnnUsermodel()
  load_cnnTweetmodel()

	app.run(debug=True, port=5000,threaded=True)
