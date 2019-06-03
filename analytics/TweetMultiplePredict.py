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



def predict(inputFile, outputFile, parameterOut,type):
    if type == 's':
        with open('SVMtweetOnly.pickle','rb') as f:
            load_model=pickle.load(f)
    if type =='r':
        with open('RFtweetOnly.pickle','rb') as f:
            load_model=pickle.load(f)
    if type =='k':
        with open('KNNtweetOnly.pickle','rb') as f:
            load_model=pickle.load(f)
    if type =='m':
        with open('MLPtweetOnly.pickle','rb') as f:
            load_model=pickle.load(f)

    bDataFrame = pd.read_csv(inputFile).fillna('')
    parameterIn = ['tweet']
    X_test = bDataFrame[parameterIn]
    bTest = load_model.predict(X_test)  # apply the model to the test data
    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile, mode='a',header=False)

if __name__ == '__main__':
    inputFile=sys.argv[1]
    outputFile=sys.argv[2]
    df = pd.read_csv(inputFile)
    df = df.fillna('')
    classifier= sys.argv[3]
    if classifier =='s':
        predict(inputFile,outputFile,'newcategory','s')
    if classifier =='r':
        predict(inputFile,outputFile,'newcategory','r')
    if classifier =='k':
        predict(inputFile,outputFile,'newcategory','k')
    if classifier =='m':
        predict(inputFile,outputFile,'newcategory','m')

    #run_model(X, y, pipeline, 'dataCombined.csv',  'dataCombinedUserLabelled.csv', 'user_category' )
    


