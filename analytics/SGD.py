
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, log_loss

from helperClasses import *

import pickle



trainsource="CombinedTaggedTweets.csv"
test="CombinedTaggedTweets"
df = pd.read_csv(trainsource)
df = df.fillna('')
#train_test_set = df.sample(frac=0.99, random_state=100)
parameterIn = ['user_description', 'user_verified', 'user_screen_name', 'influence_ratio']
X = df[ parameterIn ]
y = df['user_category']
f1_list=[]

def featureExtraction(inputFile):
	df = pd.read_csv(inputFile)
	name_extractor = np.asarray(df['user_description']).astype(str)
	vect = CountVectorizer(ngram_range=(1,6), analyzer='char')  
	tfidf = TfidfTransformer()
	features=tfidf.fit_transform(vect.fit_transform(name_extractor))
	print(features)
	return features

def train_model(trainsource):
	clf = SGDClassifier()
	features= featureExtraction(trainsource)
	X_train, X_test, y_train, y_true = train_test_split(features, y, test_size=0.2)
	clf.fit(X_train,y_train)
	with open('SGDmodel.pickle','wb') as f:
		pickle.dump(clf,f)
	y_test=clf.predict(X_test)
	score = accuracy_score(y_test, y_true)
	f1score=f1_score(y_test,y_true, average=None)
	f1_list.append(f1score) 
	print(score)

def retune(inputFile):

	features= featureExtraction(inputFile)

	with open('SGDmodel.pickle','rb') as f:
		load_model=pickle.load(f)
	ylabel=load_model.predict(features)

	X_train, X_test, y_train, y_true = train_test_split(features, ylabel, test_size=0.2)
	print(X_train.shape)
	load_model.partial_fit(X_train, y_train)
	y_test=load_model.predict(X_test)
	print(y_test)
	#proba=load_model.predict_proba(X_test)
	#logloss=log_loss(y_true,proba)
	score = accuracy_score(y_test, y_true) 
	f1=f1_score(y_test, y_true, average=None)
	print(f1)
	value=(np.mean(f1_list))* 0.95
	if np.mean(f1) > value :
		clf=load_model
	with open('SGDmodel.pickle','wb') as f:
		pickle.dump(clf,f)
		f1_list.append(np.mean(f1))

	print(score)

train_model(trainsource)
retune(trainsource)

