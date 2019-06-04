from flask import Flask
from flask import request
import pandas as pd 
import numpy as np 
import pickle
from helperClasses import *

app = Flask(__name__)

@app.route('/svmuser', methods=['GET','POST'])
def SVMuser():
	model = open("SVMUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['user_description']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/svmtweet', methods=['GET','POST'])
def SVMtweet():
	model = open("SVMTweetOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/knnuser', methods=['GET','POST'])
def KNNuser():
	model = open("KNNUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['user_description']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/knntweet', methods=['GET','POST'])
def KNNtweet():
	model = open("KNNTweetOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/rfuser', methods=['GET','POST'])
def RFuser():
	model = open("RFUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['user_description']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/rftweet', methods=['GET','POST'])
def RFtweet():
	model = open("RFTweetOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/mlpuser', methods=['GET','POST'])
def MLPuser():
	model = open("MLPUserOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['user_description']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result

@app.route('/mlptweet', methods=['GET','POST'])
def MLPtweet():
	model = open("MLPTweetOnly.pickle","rb")
	clf = pickle.load(model)
	#modify it for the data stream
	df= pd.read_csv("test1.csv").fillna('')
	parameterIn=['tweet']
	X_test = df[parameterIn]
	bTest = clf.predict(X_test)
	result=str(bTest)
	return result


if __name__ == '__main__':
    app.run(debug=True, port=33507)