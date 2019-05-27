# coding:utf-8
import os
import codecs
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.layers import Input, GRU, BatchNormalization, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from bert.extract_feature import BertVector
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class BertClassification(object):
	def __init__(self,
				 nb_classes=5,
				 dense_dim=256,
				 max_len=60,
				 batch_size=256,
				 epochs=5,
				 weights_file_path="./model/weights_tweet.h5"):
		self.nb_classes = nb_classes
		self.dense_dim = dense_dim
		self.max_len = max_len
		self.batch_size = batch_size
		self.epochs = epochs
		self.weights_file_path = weights_file_path

		self.nb_samples = 25000 # 样本数
		self.bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=self.max_len)

	def text2bert(self, text):
		vec = self.bert_model.encode([text])
		return vec["encodes"][0]

	def data_format(self, lines):
		X, y = [], []
		for line in lines:
			line = line.strip().split("\t")
			label = int(line[0])
			content = line[1]
			vec = self.text2bert(content)
			X.append(vec)
			y.append(label)
		X = np.array(X)
		y = np_utils.to_categorical(np.asarray(y), num_classes=self.nb_classes)
		return X, y

	def create_model(self):
		x_in = Input(shape=(768, ))
		x_out = Dense(self.dense_dim, activation="relu")(x_in)
		x_out = BatchNormalization()(x_out)
		x_out = Dense(self.nb_classes, activation="softmax")(x_out)
		model = Model(inputs=x_in, outputs=x_out)
		return model

	def train(self):
		model = self.create_model()
		model.compile(loss='categorical_crossentropy',
					  optimizer=Adam(),
					  metrics=['accuracy'])

		checkpoint = ModelCheckpoint(self.weights_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		pf = pd.read_csv('test1.csv')
		X=pf['tweet']
		Y=pf['category']
		tweet=[]
		for x in X:
			vec=self.text2bert(x)
			tweet.append(vec)
		tweet=np.array(tweet)
		le = preprocessing.LabelEncoder()
		le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
		Y=le.transform(Y)
		labels=np_utils.to_categorical(np.asarray(Y), num_classes=self.nb_classes)
		x_train, x_test, y_train, y_test = train_test_split(tweet, labels, test_size=0.1)
		model.fit(x_train,y_train,
							#steps_per_epoch=int(self.nb_samples/self.batch_size)+1,
							epochs=self.epochs,
							verbose=1,
							validation_data=(x_test, y_test),
							callbacks=[checkpoint]
							)


if __name__ == "__main__":
	train = len(sys.argv) == 1
	le = preprocessing.LabelEncoder()
	le.fit(["educational", "personal", "unrelated", "promotional","fundraising"])
	bc = BertClassification()
	if train:
		bc.train()
	else:
		model = load_model("./model/weights_tweet.h5")
		TEST_PROCESSED_FILE = sys.argv[1]
		md = pd.read_csv(TEST_PROCESSED_FILE)
		X=md['tweet']
		test_tweets=[]
		for x in X:
			vec=bc.text2bert(x)
			test_tweets.append(vec)
		test_tweets=np.array(test_tweets)
		predictions = model.predict(test_tweets, batch_size=128, verbose=1)
		results =  np.argmax(predictions, axis=1).astype(int)
		mDataFrame=pd.read_csv('test1.csv').fillna('')
		mDataFrame['new']=le.inverse_transform(results)
		mDataFrame.to_csv(TEST_PROCESSED_FILE)






