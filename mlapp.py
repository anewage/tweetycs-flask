from flask import Flask
from flask import request,jsonify
import pandas as pd 
import numpy as np 
import pickle
from helperClasses import *
import json
from pandas.io.json import json_normalize

app = Flask(__name__)



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


if __name__ == '__main__':
	load_svmTweetmodel()
	load_svmUsermodel()
	load_knnTweetmodel()
	load_knnUsermodel()
	load_rfTweetmodel()
	load_rfUsermodel()
	load_mlpTweetmodel()
	load_mlpUsermodel()

	app.run(debug=True, port=33508,threaded=True)
