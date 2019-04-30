
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
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




def featureExtraction(inputFile):
	df = pd.read_csv(inputFile).fillna('')
	y = df['user_category']
	name_extractor = np.asarray(df['user_description']).astype(str)
	#vect = CountVectorizer(ngram_range=(1,6), analyzer='char')  
	#tfidf = TfidfTransformer()
	#features=tfidf.fit_transform(vect.fit_transform(name_extractor))
	ha=HashingVectorizer()
	features=ha.fit_transform(name_extractor)
	return features

def featurePartial(inputFile):
	df = pd.read_csv(inputFile).fillna('')
	name_extractor = np.asarray(df['user_description']).astype(str)
	ha=HashingVectorizer()
	hashing=ha.partial_fit(name_extractor)
	features=hashing.transform(name_extractor)
	return features
#SVM
def train_model_svm(trainsource):
	clf = SGDClassifier(max_iter=1000,n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',power_t=0.5, random_state=None, shuffle=True, tol=0.001,verbose=0, warm_start=False)
	features= featureExtraction(trainsource)
	X_train, X_test, y_train, y_true = train_test_split(features, y, test_size=0.1)
	clf.fit(X_train,y_train)
	with open('SGDmodelsvm.pickle','wb') as f:
		pickle.dump(clf,f)
	y_test=clf.predict(X_test)
	score = accuracy_score(y_test, y_true)
	f1score=f1_score(y_test,y_true, average='macro')
	f1_list.append(f1score) 
	print('train svm', score)

#logistic Regression
def train_model_lr(trainsource):
	clf = SGDClassifier(loss='log', max_iter=1000,n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='none',power_t=0.5, random_state=None, shuffle=True, tol=0.001,verbose=0, warm_start=False)
	features= featureExtraction(trainsource)
	X_train, X_test, y_train, y_true = train_test_split(features, y, test_size=0.2)
	clf.fit(X_train,y_train)
	with open('SGDmodellr.pickle','wb') as f:
		pickle.dump(clf,f)
	y_test=clf.predict(X_test)
	score = accuracy_score(y_test, y_true)
	f1score=f1_score(y_test,y_true, average='macro')
	f1_list2.append(f1score) 
	print('train LR', score)


def retune_svm(inputFile):

	features= featurePartial(inputFile)

	with open('SGDmodelsvm.pickle','rb') as f:
		load_model=pickle.load(f)
	ylabel=load_model.predict(features)

	X_train, X_test, y_train, y_true = train_test_split(features, ylabel, test_size=0.2)
	load_model.partial_fit(X_train, y_train)
	y_test=load_model.predict(X_test)
	#proba=load_model.predict_proba(X_test)
	#logloss=log_loss(y_true,proba)
	score = accuracy_score(y_test, y_true) 
	f1=f1_score(y_test, y_true, average='macro')
	value=(np.mean(f1_list))* 0.95
	if f1 > value :
		clf=load_model
	with open('SGDmodelsvm.pickle','wb') as f:
		pickle.dump(clf,f)
		f1_list.append(np.mean(f1))

	print('svm', score)

def retune_lr(inputFile):

	features= featurePartial(inputFile)

	with open('SGDmodellr.pickle','rb') as f:
		load_model=pickle.load(f)
	ylabel=load_model.predict(features)

	X_train, X_test, y_train, y_true = train_test_split(features, ylabel, test_size=0.2)
	load_model.partial_fit(X_train, y_train)
	y_test=load_model.predict(X_test)
	#proba=load_model.predict_proba(X_test)
	#logloss=log_loss(y_true,proba)
	score = accuracy_score(y_test, y_true) 
	f1=f1_score(y_test, y_true, average='macro')
	value=(np.mean(f1_list2))* 0.95
	if f1 > value :
		clf=load_model
	with open('SGDmodellr.pickle','wb') as f:
		pickle.dump(clf,f)
		f1_list2.append(np.mean(f1))

	print('lr', score)

if __name__ == '__main__':



	trainsource="test1.csv"
	df = pd.read_csv(trainsource).fillna('')

	#df = df.sample(frac=0.99, random_state=100)
	parameterIn = ['user_description', 'user_verified', 'user_screen_name', 'influence_ratio']
	y = df['user_category']
	f1_list=[]
	f1_list2=[]

    #svm
	train_model_svm(trainsource)
	retune_svm('test2.csv')
    #logistic regression
	train_model_lr(trainsource)
	retune_lr('test2.csv')

