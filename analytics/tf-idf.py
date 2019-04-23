
    
#!/usr/bin/python
# coding=utf-8
import sys,codecs
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer


def dataPrepos(text, stopkey):
    l = []
    words = text.split()
    for i in words:
        if i not in stopkey:  # stopwords
            l.append(i)
    return l

def stem(text):
    stemmed_tweet = []
    ps = PorterStemmer()
    words = word_tokenize(text)
    for w in words:
        p = ps.stem(w)
        stemmed_tweet.append(p)
    return(' '.join(stemmed_tweet))

def lemmatize(text):
    lem_tweet = []
    lem = WordNetLemmatizer()
    words = word_tokenize(text)
    for w in words:
        l = lem.lemmatize(w)
        lem_tweet.append(l)
    return(' '.join(lem_tweet))

# tf-idf
def getKeywords_tfidf(data,stopkey,topK):

    idList =data['id']
    tweet = data['text']
    corpus=[]
    for t in tweet:
    #     processed_tweet.append(t)
    # textp=' '.join(processed_tweet)
        words = t.lower().split()
        meaningful_words = [w for w in words if not w in stopkey]
        textf=' '.join(meaningful_words)
        textm=stem(textf)
        corpus.append(textm) 
    #textm=lemmatize(textf)

    #construct the vector
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) 
    #Calculate the TF-IDF
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # Get the keyword
    word = vectorizer.get_feature_names()
    # To get the keyword
    weight = tfidf.toarray()

    # Print the weight
    ids, texts, keys = [], [], []
    for i in range(len(weight)):
        ids.append(idList[i])
        df_word,df_weight = [],[] #  All the words and weights
        for j in range(len(word)):
            # print (word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word,columns=['word'])
        df_weight = pd.DataFrame(df_weight,columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) #
        word_weight = word_weight.sort_values(by="weight",ascending = False) # sort
        #print(word_weight)
        keyword = np.array(word_weight['word']) 
        word_split = [keyword[x] for x in range(0,topK)] 

    frequency={}
    for word in textm.split():
        if word in word_split:
            if word not in frequency:
                frequency[word]=1
            else:
                frequency[word]+=1
    data_sorted = {k: v for k, v in sorted(frequency.items(), key=lambda x: x[1], reverse=True)}
    print(data_sorted)
    df = pd.DataFrame.from_dict(data_sorted, orient='index')
    return df


def main():
    csvfile = sys.argv[1]
    number = int(sys.argv[2])
    # read the file
    data = pd.read_csv(csvfile).fillna('')
    # Get the stopwords
    stopkey = [w.strip() for w in codecs.open('stopwords.txt', 'r').readlines()]
    # extract the tf-idf keyword
    result = getKeywords_tfidf(data,stopkey,number)
    result.to_csv("keys_TFIDF.csv")

if __name__ == '__main__':
    main()
