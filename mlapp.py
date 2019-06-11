from flask import Flask
from flask import request,jsonify
import pandas as pd 
import numpy as np 
import pickle
from helperClasses import *
import json
from pandas.io.json import json_normalize

app = Flask(__name__)


@app.route('/svmtweet', methods=['GET','POST'])
def SVMtweet():
	model = open("models/SVMtweetOnly.pickle","rb")
	clf = pickle.load(model)
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=bTest
	result=result[0]
	return jsonify(theme=result)

@app.route('/svmuser', methods=['GET','POST'])
def SVMuser():
	model = open("models/SVMUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	parameterIn=['user_description']
	test = df[parameterIn]
	bTest=clf.predict(test)
	result=bTest
	result=result[0]
	return jsonify(group=result)

@app.route('/knnuser', methods=['GET','POST'])
def KNNuser():
	model = open("models/KNNUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	parameterIn=['user_description']
	test = df[parameterIn]
	bTest=clf.predict(test)
	result=bTest
	result=result[0]
	return jsonify(group=result)

@app.route('/knntweet', methods=['GET','POST'])
def KNNtweet():
	model = open("models/KNNTweetOnly.pickle","rb")
	clf = pickle.load(model)
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=bTest
	result=result[0]
	return jsonify(theme=result)

@app.route('/rfuser', methods=['GET','POST'])
def RFuser():
	model = open("models/RFUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	parameterIn=['user_description']
	test = df[parameterIn]
	bTest=clf.predict(test)
	result=bTest
	result=result[0]
	return jsonify(group=result)

@app.route('/rftweet', methods=['GET','POST'])
def RFtweet():
	model = open("models/RFTweetOnly.pickle","rb")
	clf = pickle.load(model)
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=bTest
	result=result[0]
	return jsonify(theme=result)

@app.route('/mlpuser', methods=['GET','POST'])
def MLPuser():
	model = open("models/MLPUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	parameterIn=['user_description']
	test = df[parameterIn]
	bTest=clf.predict(test)
	result=bTest
	result=result[0]
	return jsonify(group=result)


@app.route('/mlptweet', methods=['GET','POST'])
def MLPtweet():
	model = open("models/MLPTweetOnly.pickle","rb")
	clf = pickle.load(model)
	data=request.get_json(force=True)
	df = pd.DataFrame.from_dict(json_normalize(data))
	#data=pd.read_json(data)
	#json_normalize(data['text'])
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=bTest
	result=result[0]
	return jsonify(theme=result)


if __name__ == '__main__':
    app.run(debug=True, port=33508)
