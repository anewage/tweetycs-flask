import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Embedding, Flatten
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import pandas as pd
import os
from collections import Counter
import tensorflow as  tf

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Performs classification using CNN.

TRAIN_PROCESSED_FILE = 'test1.csv'
GLOVE_FILE = 'glove.twitter.27B.200d.txt'
dim = 200


def get_glove_vectors(vocab):

	#Extracts glove vectors from seed file only for words present in vocab.
	print ('Looking for GLOVE seeds')
	glove_vectors = {}
	found = 0
	with open(GLOVE_FILE, 'r') as glove_file:
		for i, line in enumerate(glove_file):
			write_status(i + 1, 0)
			tokens = line.strip().split()
			word = tokens[0]
			if vocab.get(word):
				vector = [float(e) for e in tokens[1:]]
				glove_vectors[word] = np.array(vector)
				found += 1
	print ('\n')
	return glove_vectors

def write_status(i, total):
	''' Writes status of a process to console '''
	sys.stdout.write('\r')
	sys.stdout.write('Processing %d/%d' % (i, total))
	sys.stdout.flush()

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


def process_tweets(csv_file, test_file=True):

	#Generates training X, y pairs.

	tweets = []
	labels = []
	print ('Generating feature vectors')
	bDataFrame = pd.read_csv(csv_file).fillna('')
	tweet = bDataFrame['tweet']
	senti = bDataFrame['category']
	total = len(tweet)
	for t in tweet:
		feature_vector = get_feature_vector(t)
		if test_file:
			tweets.append(feature_vector)
		else:
			tweets.append(feature_vector)
	if test_file:
		labels = []
	else:
		for m in senti:
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


if __name__ == '__main__':
	train = len(sys.argv) == 1
	np.random.seed(1337)
	vocab_size = 90000
	max_length = 140
	mDataFrame = pd.read_csv(TRAIN_PROCESSED_FILE).fillna('')
	tweetm = mDataFrame['tweet']
	vocab = top_n_words(tweetm,vocab_size,shift=1)
	glove_vectors = get_glove_vectors(vocab)
	tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
	# Create and embedding matrix
	embedding_Matrix = np.random.randn(vocab_size + 1, dim) * 0.01
	for word, i in vocab.items():
		glove_vector = glove_vectors.get(word)
		if glove_vector is not None:
			embedding_Matrix[i] = glove_vector
	tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
	shuffled_indices = np.random.permutation(tweets.shape[0])
	tweets = tweets[shuffled_indices]
	labels = labels[shuffled_indices]
	le = preprocessing.LabelEncoder()
	le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
	labels=le.transform(labels)
	labels = to_categorical(labels, 5)
	if train:
		model = Sequential()  # or Graph or whatever
		model.add(Embedding(input_dim=vocab_size+1,
						output_dim=200,
						mask_zero=True,
						weights=[embedding_Matrix],
						input_length=max_length))  # Adding Input Length
		model.add(LSTM(output_dim=128, activation='tanh',use_bias=True))
		model.add(Dropout(0.5))
		model.add(Dense(5, activation='softmax',use_bias=False)) # Dense=>全连接层,输出维度=3
		model.add(BatchNormalization())
		model.add(Activation('softmax'))

		print ('Compiling the Model...')
		model.compile(loss='categorical_crossentropy',
				  optimizer='adam',metrics=['accuracy'])
		print(model.summary())
		filepath = "LstmTweet.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode='max')
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
		model.fit(tweets, labels, batch_size=32, epochs=8, validation_split=0.1, shuffle=True, callbacks=[checkpoint, reduce_lr], verbose=1)
	else:
		model = load_model("LstmTweet.hdf5")
		TEST_PROCESSED_FILE = sys.argv[1]
		print (model.summary())
		test_tweets, _ = process_tweets(TEST_PROCESSED_FILE, test_file=True)
		test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
		predictions = model.predict(test_tweets, batch_size=128, verbose=1)
		results =  np.argmax(predictions, axis=1).astype(int)
		mDataFrame=pd.read_csv(TEST_PROCESSED_FILE).fillna('')
		mDataFrame['newTweetCategory']=le.inverse_transform(results)
		mDataFrame.to_csv(TEST_PROCESSED_FILE)