import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
from operator import and_, or_

import os

def createFilePath(filename):
    docLoc = os.path.abspath(os.curdir) + '\data\\' + filename
    return docLoc


# COLUMN SELECTION
class TextExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be string
        return np.asarray(df[self.column_name]).astype(str)

    def fit(self, *_):
        return self

class ValueExtractor(BaseEstimator, TransformerMixin):
    """Adapted from code by @zacstewart
       https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
       Also see Zac Stewart's excellent blogpost on pipelines:
       http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
       """

    def __init__(self, column_name):
        self.column_name = column_name

    def value_to_float(self, name):
        """Helper code to convert string to a float"""
        #return np.mean([len(word) for word in name.split()])
        return float(name)

    def transform(self, df):
        """The workhorse of this feature extractor"""
        return df[self.column_name].apply(self.value_to_float)

    #def transform(self, df):
    #    # select the relevant column and return it as a numpy array
        # set the array type to be string
   #     return np.asarray(df[self.column_name])#.astype(float)

    def fit(self, *_):
        return self


class Apply(BaseEstimator, TransformerMixin):
    """Applies a function f element-wise to the numpy array
    """

    def __init__(self, fn):
        self.fn = np.vectorize(fn)

    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self
# we already imported FeatureUnion earlier, so here goes

dictionaryOfStuff = {
    "news": 1,
    "journal": 1,
    "paper": 1,
    "advertisement": 1,
    "official": 1,
    "market": 1,
    "register": 1,
    "source": 1,
    "newspaper":1,
}
class AllDictionaryWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dictionary = dictionaryOfStuff
        self.fn = np.vectorize(self.all_words_in_dict)

    def all_words_in_dict(self, s):
        return reduce(or_, [word.lower() in self.dictionary
                             for word in s.split()])

    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self

class CountOfKeywords(BaseEstimator, TransformerMixin):
    def __init__(self, dictionaryToUse):
        self.dictionary = dictionaryToUse
        self.fn = np.vectorize(self.all_words_in_dict)

    def all_words_in_dict(self, s):
        return reduce(or_, [word.lower() in self.dictionary
                             for word in s.split()])

    def count_words_in_dict(self, s):
        return np.sum([word.lower() in self.dictionary
                             for word in s.split()])

    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self

class BooleanExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be string
        return np.asarray(df[self.column_name]).astype(np.bool)

    def fit(self, *_):
        return self

keyWords = ['health', 'news',  'headlines',
 'bot', 'benefit', 'organization', 'diet', 'charity',
 'products', 'business', 'provider', 'dentistry',
 'booking', 'volunteer','industry', 'fitness',
 'consumer', 'charitable', 'information', 'business',
 'breaking', 'healthcare', 'university', 'sports', 'publication','journal',
 'reports', 'research', 'trials', 'reviews', 'radio', 'official', 'updates',
 'developments', 'world', "news","paper", "official",    "source",    "newspaper",    "journal",    "advertisement"
]
#include health issues
keyWords = list(set(keyWords))
dictionaryOfKeywords = {}
for i in range(len(keyWords)):
    dictionaryOfKeywords[keyWords[i]] = i
