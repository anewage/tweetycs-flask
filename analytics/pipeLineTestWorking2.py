import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC



from helperClasses import *
# Desc char:
# 0.877104377104
# 0.857523553163
# Tfidf:
# 0.890572390572
# 0.868627187079
# User_Name 2
# 0.811447811448
# 0.798317631225
# Prev w/ follower:
# 0.794612794613
# 0.798398384926
# Prev w/ verified:
# 0.799663299663
# 0.797537012113



#
# Desc char:
# 0.796703296703
# 0.747224669604
# Tfidf:
# 0.813186813187
# 0.761674008811
# User_Name 2
# 0.802197802198
# 0.746431718062
# Prev w/ follower:
# 0.802197802198
# 0.761057268722


#SECOND RUN
# Desc char:
# 0.763736263736
# 0.748810572687
# Tfidf:
# 0.774725274725
# 0.761938325991
# User_Name 2
# 0.78021978022
# 0.752511013216
# Prev w/ follower:
# 0.754713656388
# 0.796703296703

#With 3000




# #With 2720

# Desc char:
# 0.86827458256
# 0.875296735905
# Tfidf:
# 0.879406307978
# 0.886928783383
# User_Name 2
# 0.877551020408
# 0.880296735905
# Prev w/ follower:
# 0.858998144712
# 0.86821958457
# Prev w/ verified:
# 0.857142857143
# 0.866261127596

# Desc char:
# 0.877551020408
# 0.877359050445
# Tfidf:
# 0.881261595547
# 0.888738872404
# User_Name 2
# 0.884972170686
# 0.879376854599
# Prev w/ follower:
# 0.857142857143
# 0.866795252226
# Prev w/ verified:
# 0.857142857143
# 0.868991097923
# Desc char:
# 0.774725274725
# 0.746696035242
# Tfidf:
# 0.796703296703
# 0.757885462555
# User_Name 2
# 0.763736263736
# 0.753480176211
# Prev w/ follower:
# 0.785714285714
# 0.753127753304
# Prev w/ verified:
# 0.802197802198
# 0.756475770925
#os.chdir("..")


#df = pd.read_csv('testFile.csv')
#df = pd.read_csv('tweetLabelledUserLabelledWith3.csv')


df = pd.read_csv('CombinedTaggedTweets.csv')
df = df.fillna('') #this doesn't really help us
train_test_set = df.sample(frac=0.99, random_state=10)
X = train_test_set[ ['user_description', 'user_verified', 'user_screen_name', 'influence_ratio'] ]
y = train_test_set['user_category']
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2)
#So now the datat we are using is the same for all runs
def run_experiment(pipeline):
    model = pipeline.fit(X_train, y_train)  # train the classifier
    y_test = model.predict(X_test)          # apply the model to the test data
    score = accuracy_score(y_test, y_true)  # compare the results to the gold standard
    print (score)
def run_experimentFull(X, y, pipeline, num_expts=100):
    scores = list()
    for i in range(num_expts):
        X_train, X_test, y_train, y_true = train_test_split(X, y)
        model = pipeline.fit(X_train, y_train)  # train the classifier
        y_test = model.predict(X_test)          # apply the model to the test data
        score = accuracy_score(y_test, y_true)  # compare the results to the gold standard
        scores.append(score)

    print (sum(scores) / num_expts)

vect = CountVectorizer(ngram_range=(1,5), analyzer='char')
clf = LinearSVC()

#FIRST: Just the description
pipeline = Pipeline([
    ('name_extractor', TextExtractor('user_description')),  # extract names from df
    ('vect', vect),  # extract ngrams from roadnames
    ('clf' , clf),   # feed the output through a classifier
])
print('Desc char: ')
run_experiment(pipeline)
run_experimentFull(X,y, pipeline)

#FIRST HALF: Just the description but as words not characters
#vect = CountVectorizer(analyzer='word')
pipeline = Pipeline([
    ('name_extractor', TextExtractor('user_description')),  # extract names from df
    ('vect', vect),  # extract ngrams from roadnames
    ('tfidf', TfidfTransformer() ),
    ('clf' , clf),   # feed the output through a classifier
])
print('Tfidf: ')
run_experiment(pipeline)
run_experimentFull(X,y, pipeline)


text_pipeline = Pipeline([
    ('name_extractor', TextExtractor('user_description')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', vect),  # extract words from description
        ('count_news_words', CountOfKeywords(dictionaryOfStuff)),
    ])),
    ('tfidf', TfidfTransformer()),
])
name_pipeline = Pipeline([
    ('name_extractor', TextExtractor('user_screen_name')), # extract names from df
    ('text_features', FeatureUnion([
        ('vect', CountVectorizer(ngram_range=(3,7), analyzer='char') ),
       # ('num_words', Apply(lambda s: len(s))), # length of string
       ('count_news_words', CountVectorizer(ngram_range=(3,13), analyzer='char', vocabulary=dictionaryOfKeywords) )
        #('all_dictionary_words', AllDictionaryWords()),
    ])),
])
pipeline = Pipeline([
    ('all_features', FeatureUnion([
        ('text_pipeline', text_pipeline), # all text features
        ('name_pipeline', name_pipeline),
    #    ('tfidf', TfidfTransformer()),
    ])),
    ('clf' , clf),   # feed the output through a classifier
])
print('User_Name 2 ')
run_experiment(pipeline)
run_experimentFull(X,y, pipeline)



#Following Follower Ratio
follower_pipeline = Pipeline([
  ('val_feature', ValueExtractor('influence_ratio')),
 # ('name_extractor', TextExtractor('influence_ratio')), # extract names from df
  ('identity', Apply(lambda x: x)), # this is a bit silly but we need to do the transform and this was the easiest way to do it
])

pipeline = Pipeline([
    ('all_features', FeatureUnion([
        ('text_pipeline', text_pipeline), # all text features
        ('name_pipeline', name_pipeline),
    #    ('verified_pipeline', verified_pipeline),
        ('follower_pipeline', follower_pipeline)
    ])),
    ('clf' , clf),   # feed the output through a classifier
])
print('Prev w/ follower: ')
run_experiment(pipeline)
run_experimentFull(X,y, pipeline)

verified_pipeline = Pipeline([
    ('ver_feature', ValueExtractor('user_verified')),
    ('identity', Apply(lambda x: x)),
    # this is a bit silly but we need to do the transform and this was the easiest way to do it
])

pipeline = Pipeline([
    ('all_features', FeatureUnion([
        ('text_pipeline', text_pipeline), # all text features
        ('name_pipeline', name_pipeline),
        ('verified_pipeline', verified_pipeline),
        ('follower_pipeline', follower_pipeline)
    ])),
    ('clf' , clf),   # feed the output through a classifier
])
print('Prev w/ verified: ')
run_experiment(pipeline)
run_experimentFull(X,y, pipeline)



parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}

#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
