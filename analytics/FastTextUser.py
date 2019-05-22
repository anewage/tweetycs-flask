# coding=utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
import pandas as pd
from fast_text import FastText
from collections import Counter
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Embedding,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 50000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

def get_feature_vector(tweet):

   # Generates a feature vector for each tweet where each word is represented by integer index based on rank in vocabulary.

    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector

def process_tweets(X, y, test_file=True):

    #Generates training X, y pairs.

    tweets = []
    labels = []
    total = len(X)
    for t in X:
        feature_vector = get_feature_vector(t)
        if test_file:
            tweets.append(feature_vector)
        else:
            tweets.append(feature_vector)
    for m in y:
        labels.append(m)
    print ('\n')
    return tweets, np.array(labels)

def top_n_words(tweetm,n,shift=0):

    freq_dict = {}
    tweet=' '.join(tweetm)
    words = tweet.split()
    for word in words:
        if freq_dict.get(word):
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    freq_dist = Counter(freq_dict)
    most_common = freq_dist.most_common(n)
    words = {p[0]: i + shift for i, p in enumerate(most_common)}
    return words


print('Loading data...')
df=pd.read_csv('test1.csv').fillna('')
X=df['user_description']
y=df['user_category']
print(len(y))
vocab_size=90000
vocab=top_n_words(X,vocab_size,shift=1)
tweet,labels=process_tweets(X,y)
print(len(tweet))
print(len(labels))
le = preprocessing.LabelEncoder()
le.fit(["public","interest groups","media","businesses","celebrities","official agencies"])
labels=le.transform(labels)
print(labels)
labels = to_categorical(labels, 6)
x_train, x_test, y_train, y_test = train_test_split(tweet, labels, test_size=0.1)
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims,input_length=maxlen))
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print('Train...')
filepath='FastText.hdf5'
#early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[checkpoint, reduce_lr],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
print(result)