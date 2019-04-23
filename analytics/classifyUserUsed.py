import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from helperClasses import *
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

trainsource='CombinedTaggedTweets.csv'
df = pd.read_csv(trainsource)
df = df.fillna('')
train_test_set = df.sample(frac=0.99, random_state=100)
parameterIn = ['user_description', 'user_verified', 'user_screen_name', 'influence_ratio']
X = train_test_set[ parameterIn ]
y = train_test_set['user_category']

vect = CountVectorizer(ngram_range=(1,6), analyzer='char')                                                            
clf = LinearSVC()
pipeline = Pipeline([
    ('name_extractor', TextExtractor('user_description')),  # extract names from df
    ('vect', vect),  # extract ngrams from roadnames
    ('tfidf', TfidfTransformer() ),
    ('clf' , clf),   # feed the output through a classifier
])
print('Tfidf: ')

def run_experiment(X, y, pipeline, num_expts=10):
    scores = list()
    for i in range(num_expts):
        X_train, X_test, y_train, y_true = train_test_split(X, y)
        model = pipeline.fit(X_train, y_train)  # train the classifier
        y_test = model.predict(X_test)          # apply the model to the test data
        score = accuracy_score(y_test, y_true)  # compare the results to the gold standard
        scores.append(score)
        #print( classification_report(y_true, y_test) )
        #print( confusion_matrix(y_true, y_test) )
    print(sum(scores) / num_expts)

def run_model(X_train, y_train, pipeline, inputFile,  outputFile, parameterOut):
    bDataFrame = pd.read_csv(inputFile).fillna('')
    X_test = bDataFrame[parameterIn]
    model = pipeline.fit(X_train, y_train)  # train the classifier
    bTest = model.predict(X_test)  # apply the model to the test data

    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile)

def run_model_self(X_train, y_train, pipeline, inputFile,  outputFile, parameterOut):
    bDataFrame = pd.read_csv(inputFile).fillna('')
    X_test = bDataFrame[parameterIn]
    model = pipeline.fit(X_train, y_train)  # train the classifier
    bTest = model.predict(X_test)  # apply the model to the test data
    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile, mode='a',header=False, sep=',', index=False)

def self_training(X_train, y_train, pipeline, inputFile,parameterOut):
    run_model_self(X_train, y_train, pipeline,inputFile,trainsource, parameterOut)


def predict(X_train, y_train,inputFile, outputFile, parameterOut):

    bDataFrame = pd.read_csv(inputFile).fillna('')
    bTest = model.predict(X_test)  # apply the model to the test data
    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile, mode='a',header=False)

run_experiment(X,y,pipeline)
#run_model(X, y, pipeline, 'dataCombined.csv',  'dataCombinedUserLabelled.csv', 'user_category' )
#self_training(X, y, pipeline, 'dataCombined.csv','user_category' )


