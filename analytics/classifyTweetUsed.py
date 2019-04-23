import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from helperClasses import *


#EXAMPLE
# Tweet:
# 0.810845588235
# Count new words
# 0.807720588235
# Added User Verified
# 0.800735294118

#STEP 1: LOAD DATA
#*****************************************************************************************
df = pd.read_csv('CombinedTaggedTweets.csv').fillna('')
X = df[['tweet', 'user_description','user_verified', 'user_category']]
y = df['category']
print('**************************************************************')

#****************************************************************************************************

#STEP 2: TRAINING A CLASSIFIER
def run_experiment(X, y, pipeline, num_expts=10):
    scores = list()
    for i in range(num_expts):
        X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2)
        model = pipeline.fit(X_train, y_train)  # train the classifier
        y_test = model.predict(X_test)          # apply the model to the test data
        score = accuracy_score(y_test, y_true)  # compare the results to the gold standard
        scores.append(score)
     #   print( classification_report(y_true, y_test) )
      #  print( confusion_matrix(y_true, y_test) )

    print (sum(scores) / num_expts)

#MODEL 1
vect = CountVectorizer(ngram_range=(2,10), analyzer='char')
clf = LinearSVC()
tfidf = TfidfTransformer()
tweet_pipeline = Pipeline([
    ('name_extractor', TextExtractor('tweet')),  # extract names from df
    ('vect', vect),  # extract ngrams from roadnames
    ('tfidf', tfidf),
    ('clf' , clf),   # feed the output through a classifier
])
print('Tweet: ')
run_experiment(X, y, tweet_pipeline)

#AT THIS POINT NOW LET US DEAL WITH UNLABELLED DATA FILE AND OUTPUT IT
def run_model(X_train, y_train, pipeline, inputFile, parameterIn, outputFile, parameterOut):
    bDataFrame = pd.read_csv(inputFile).fillna('')
    X_test = bDataFrame[parameterIn]
    model = pipeline.fit(X_train, y_train)  # train the classifier
    bTest = model.predict(X_test)  # apply the model to the test data

    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile)


def run_model_self(X_train, y_train, pipeline, inputFile, parameterIn, outputFile, parameterOut):
    bDataFrame = pd.read_csv(inputFile).fillna('')
    X_test = bDataFrame[parameterIn]
    model = pipeline.fit(X_train, y_train)  # train the classifier
    bTest = model.predict(X_test)  # apply the model to the test data

    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile, mode='a',header=False, sep=',', index=False)

def self_training(X_train, y_train, pipeline, inputFile,parameterOut):
    run_model_self(X_train, y_train, pipeline,inputFile,trainsource,parameterOut)


#STEP 3: RUN THE CLASSIFIER MODEL ON OUR DATA.
run_model(X, y, tweet_pipeline, 'dataCombinedUserLabelled.csv', ['tweet', 'user_description' ,'user_category'], 'dataCombinedBothClassified.csv', 'category' )



