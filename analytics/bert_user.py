# coding:utf-8
import os
import codecs
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, GRU, BatchNormalization, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from bert.extract_feature import BertVector
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下
基于bert句向量的文本分类：基于Dense的微调
"""
class BertClassification(object):
	def __init__(self,
				 nb_classes=6,
				 dense_dim=256,
				 max_len=100,
				 batch_size=128,
				 epochs=5,
				 train_corpus_path="data/sent.train",
				 test_corpus_path="data/sent.test",
				 weights_file_path="./model/weights_user.h5"):
		self.nb_classes = nb_classes
		self.dense_dim = dense_dim
		self.max_len = max_len
		self.batch_size = batch_size
		self.epochs = epochs
		self.weights_file_path = weights_file_path
		self.train_corpus_path = train_corpus_path
		self.test_corpus_path = test_corpus_path

		self.nb_samples = 25000 # 样本数
		self.bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=self.max_len)

	def text2bert(self, text):
		""" 将文本转换为bert向量  """
		vec = self.bert_model.encode([text])
		return vec["encodes"][0]

	def data_format(self, lines):
		""" 将数据转换为训练格式，输入为列表  """
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
		X=pf['user_description']
		Y=pf['user_category']
		tweet=[]
		for x in X:
			vec=self.text2bert(x)
			tweet.append(vec)
		tweet=np.array(tweet)
		le = preprocessing.LabelEncoder()
		le.fit(["public","interest groups","media","businesses","celebrities","official agencies"])
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
	bc = BertClassification()
	bc.train()




