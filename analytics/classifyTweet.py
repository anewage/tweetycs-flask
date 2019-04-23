import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from helperClasses import *

#os.chdir("..")

#STEP 1: LOAD DATA
#*****************************************************************************************
df = pd.read_csv('CombinedTaggedTweets.csv').fillna('')
X = df[['tweet', 'user_description','user_verified', 'user_category']]
y = df['category']
print('**************************************************************')

#****************************************************************************************************
#STEP 2: FORMAT THE DATA
#BUILDING A PIPELINE, use a pipeline to do both.
#This increases ngram_range 0.52 to ).58 a 6% increase.

#STEP 3: TRAINING A CLASSIFIER
def run_experiment(X, y, pipeline, num_expts=100):
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
#FIRST: Just the tweet
tweet_pipeline = Pipeline([
    ('name_extractor', TextExtractor('tweet')),  # extract names from df
    ('vect', vect),  # extract ngrams from roadnames
    ('tfidf', tfidf),
    ('clf' , clf),   # feed the output through a classifier
])
print('Tweet: ')
run_experiment(X, y, tweet_pipeline)

text_pipeline = Pipeline([
    ('name_extractor', TextExtractor('tweet')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', vect),  # extract words from description
        ('count_news_words', CountVectorizer(analyzer='word', vocabulary=dictionaryOfKeywords) )
  #ALL THROW ERRORS :(
   #     ('tfidf', tfidf ),
      #  ('len_of_desc', Apply(lambda s: len(s))), # len of string
       # ('num_words', Apply(lambda s: len(s.split()))), # #of words in length of string
        #('ave_word_length', Apply(lambda s: np.mean([len(w) for w in s.split()]))), # average word length
    ])),
])

pipeline = Pipeline([
    ('all_features', FeatureUnion([
        ('tweet_pipeline', text_pipeline), # all text features
     #   ('user_pipeline', user_pipeline),
      #  ('user_category_pipeline', user_category_pipeline),
    ])),
    ('clf' , clf),   # feed the output through a classifier
])

print('Count new words')
run_experiment(X, y, pipeline)

verified_pipeline = Pipeline([
    ('ver_feature', ValueExtractor('user_verified')),
    ('identity', Apply(lambda x: x)),
    # this is a bit silly but we need to do the transform and this was the easiest way to do it
])
pipeline = Pipeline([
    ('all_features', FeatureUnion([
        ('tweet_pipeline', text_pipeline), # all text features
        ('verified_pipeline', verified_pipeline),
      #  ('user_category_pipeline', user_category_pipeline),
    ])),
    ('clf', clf),   # feed the output through a classifier
])

print('Added User Verified')
run_experiment(X, y, pipeline)
#
# user_pipeline = Pipeline([
#     ('user_type_extractor', TextExtractor('user_category')),  # extract names from df
#     ('text_features', FeatureUnion([
#   #  ('vect', vect),  # extract ngrams from roadnames
#         ('counts', CountVectorizer())
#         ('tfidf', TfidfTransformer())
#  #   ('clf' , clf),   # feed the output through a classifier
# ])),
#     ])

#
# user_pipeline = Pipeline([
#     ('user_type_extractor', TextExtractor('user_category')),  # extract names from df
#     ('text_features', FeatureUnion([
#   #  ('vect', vect),  # extract ngrams from roadnames
#         ('counts', CountVectorizer())
#         ('tfidf', TfidfTransformer())
#  #   ('clf' , clf),   # feed the output through a classifier
# ])),
#     ])

text_pipeline = Pipeline([
    ('name_extractor', TextExtractor('tweet')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', vect),  # extract words from description
        ('count_news_words', CountVectorizer(analyzer='word', vocabulary=dictionaryOfKeywords) )
    ])),
])
#
# user_cat_pipeline = Pipeline([
#     ('name_extractor', TextExtractor('user_category')),  # extract names from df
#     ('vect', vect),  # extract ngrams from roadnames
#     ('text_features', FeatureUnion([
#         ('vect', vect),  # extract words from description
#         ('tfidf', tfidf),
#     ])),
# ])
# pipeline = Pipeline([
#     ('all_features', FeatureUnion([
#         ('tweet_pipeline', text_pipeline), # all text features
#         ('user_cat_pipeline', user_cat_pipeline),
#       #  ('user_category_pipeline', user_category_pipeline),
#     ])),
#     ('clf', clf),   # feed the output through a classifier
# ])

print('Added User Category')
#run_experiment(X, y, pipeline)

#AT THIS POINT NOW LET US DEAL WITH UNLABELLED DATA FILE AND OUTPUT IT
def run_model(X_train, y_train, pipeline, inputFile, parameterIn, outputFile, parameterOut):
    bDataFrame = pd.read_csv(inputFile).fillna('')
    X_test = bDataFrame[parameterIn]
    model = pipeline.fit(X_train, y_train)  # train the classifier
    bTest = model.predict(X_test)  # apply the model to the test data

    bDataFrame[parameterOut] = bTest.tolist()
    bDataFrame.to_csv(outputFile)


#STEP 4: RUN THE CLASSIFIER MODEL ON OUR DATA.
#run_model(X, y, tweet_pipeline, 'sampleData10KUserLabelled.csv', ['tweet', 'user_description' ,'user_category'], 'sampleData10KUserLabelledTweetLabelled.csv', 'category' )



