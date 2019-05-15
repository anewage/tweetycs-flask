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
from sklearn.neural_network import MLPClassifier
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, log_loss

from helperClasses import *

import pickle
#
# Desc char:
# 0.840067340067
# 0.85798115747
# Tfidf:
# 0.860269360269
# 0.870174966353
# User_Name 2
# 0.848484848485
# 0.866379542396
# Prev w/ follower:
# 0.843434343434
# 0.854979811575
# Prev w/ verified:
# 0.843434343434
# 0.855572005384




#vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
#clf =  RandomForestClassifier()
#pipeline = Pipeline([
#    ('name_extractor', TextExtractor('user_description')),  # extract names from df
#    ('vect', vect),  # extract ngrams from roadnames
#    ('tfidf', TfidfTransformer() ),
#    ('clf' , clf),   # feed the output through a classifier
#])
#print('Tfidf: ')

def randomForest():
    vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
    clf =  RandomForestClassifier()
    pipeline = Pipeline([
        ('name_extractor', TextExtractor('user_description')),  # extract names from df
        ('vect', vect),  # extract ngrams from roadnames
        ('tfidf', TfidfTransformer() ),
        ('clf' , clf),   # feed the output through a classifier
    ])
    print('randomForest: ')
    return pipeline

def SVM():
    vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
    clf =  LinearSVC()
    pipeline = Pipeline([
        ('name_extractor', TextExtractor('user_description')),  # extract names from df
        ('vect', vect),  # extract ngrams from roadnames
        ('tfidf', TfidfTransformer() ),
        ('clf' , clf),   # feed the output through a classifier
    ])
    print('LinearSVC: ')
    return pipeline

#0.81
def KNN():
    vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
    clf =  KNeighborsClassifier(18)
    pipeline = Pipeline([
        ('name_extractor', TextExtractor('user_description')),  # extract names from df
        ('vect', vect),  # extract ngrams from roadnames
        ('tfidf', TfidfTransformer() ),
        ('clf' , clf),   # feed the output through a classifier
    ])
    print('KNN: ')
    return pipeline

#slow
def MLP():
    vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
    clf =  MLPClassifier(activation='relu', alpha=1, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
    pipeline = Pipeline([
        ('name_extractor', TextExtractor('user_description')),  # extract names from df
        ('vect', vect),  # extract ngrams from roadnames
        ('tfidf', TfidfTransformer() ),
        ('clf' , clf),   # feed the output through a classifier
    ])
    print('MLP: ')
    return pipeline

def SGD():
    vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
    clf = SGDClassifier(
              loss='log', 
              # log as a loss gives the logistic regression
              penalty='none', 
              # l2 as a default for the linear SVM;
              fit_intercept=True, 
              shuffle=True, 
              # shuffle after each epoch
              eta0=0.001,
              learning_rate='constant',
              average=False, 
              # computes the averaged SGD weights 
              random_state=1623,
              verbose=0,
              max_iter=1000,
              warm_start=False
            );
    pipeline = Pipeline([
        ('name_extractor', TextExtractor('user_description')),  # extract names from df
        ('vect', vect),  # extract ngrams from roadnames
        ('tfidf', TfidfTransformer() ),
        ('clf' , clf),   # feed the output through a classifier
    ])
    print('SGD: ')
    return pipeline



def run_model(X, y, pipeline, inputFile, type):
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.1)
    model = pipeline.fit(X_train, y_train)  
    y_test = model.predict(X_test)          # apply the model to the test data
    score = accuracy_score(y_test, y_true)
    print("accuracy",score)
    f1=f1_score(y_test, y_true, average='macro')
    print("f1",f1)
    # train the classifier
    if type =='s':
        with open('SVMUserOnly.pickle','wb') as f:
            pickle.dump(model,f)
    if type =='r':
        with open('RFUserOnly.pickle','wb') as f:
            pickle.dump(model,f)
    if type =='k':
        with open('KNNUserOnly.pickle','wb') as f:
            pickle.dump(model,f)
    if type =='m':
        with open('MLPUserOnly.pickle','wb') as f:
            pickle.dump(model,f)


def predict(X_train, y_train,inputFile, outputFile, parameterOut):

    bDataFrame = pd.read_csv(inputFile).fillna('')
    bTest = model.predict(X_test)  # apply the model to the test data
    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile, mode='a',header=False)

if __name__ == '__main__':
    trainsource="test1.csv"
    df = pd.read_csv(trainsource)
    df = df.fillna('')
    parameterIn = ['user_description']
    X = df[parameterIn]
    y = df['user_category']
    classifier= sys.argv[1]
    if classifier =='s':
        pipeline =SVM()
        run_model(X, y, pipeline, trainsource,'s')
    if classifier =='r':
        pipeline =randomForest()
        run_model(X, y, pipeline, trainsource,'r')
    if classifier =='k':
        pipeline=KNN()
        run_model(X, y, pipeline, trainsource,'k')
    if classifier =='m':
        pipeline=MLP()
        run_model(X, y, pipeline, trainsource,'k')

