import pandas as pd
from helperClasses import *
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout,Reshape
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile



trainsource='CombinedTaggedTweets.csv'
df = pd.read_csv(trainsource)
df = df.fillna('')
#train_test_set = df.sample(frac=0.99, random_state=100)
X = df['user_description']
y = df['user_category']

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

SENTENCE_LENGTH = 26
NUM = 5000


tokenizer = Tokenizer(num_words=NUM)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=maxlen)
labels = to_categorical(np.asarray(y), 6)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

     # split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]
num_validation_samples = int(0.1 * data.shape[0])

X_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
X_test = data[-num_validation_samples:]
y_true= labels[-num_validation_samples:]



class_num=6

def get_model():
    input = Input((maxlen,))

    # Embedding part can try multichannel as same as origin paper
    embedding = Embedding(max_features, embedding_dims, input_length=maxlen)(input)
    reshape = Reshape((maxlen,embedding_dims,1))(embedding)
    convs = []
    for kernel_size in [3, 4, 5]:
        c = Conv1D(128, kernel_size, activation='relu')(embedding)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)

    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    return model


print('Build model...')
model = get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1, 
          callbacks=[early_stopping])

print('Test...')
result = model.predict(X_test)
predictions =  np.argmax(result, axis=1).astype(int)




