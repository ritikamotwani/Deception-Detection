# gensim modules
import gensim
from gensim import utils
from gensim import corpora,models
# numpy
import numpy as np
# classifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from random import shuffle
import pandas
from sklearn import model_selection, preprocessing, naive_bayes
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from keras.models import Model
from sklearn.metrics import accuracy_score
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from utilities import read_fileNames, readFilesFromSources


#create source
sources = []
datapath = './Real_Life_Trial_Data/'

#read file names from datapath
read_fileNames(sources, datapath,'Deceptive')
deceptiveCount=len(sources)
read_fileNames(sources, datapath,'Truthful')
truthCount=len(sources)-deceptiveCount

#read text files from source list
text = []
readFilesFromSources(text,sources)

#create label array corresponding to text files
labels=np.empty(len(sources))
np.concatenate((np.ones(deceptiveCount, dtype=int),np.zeros(truthCount, dtype=int)),out=labels)


# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = text
trainDF['label'] = labels

# split the dataset into training and validation datasets 
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.15)


#Tokenize the data and convert the text to sequences.
#Add padding to ensure that all the sequences have the same shape.
#There are many ways of taking the max_len and here an arbitrary length of 150 is chosen.
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# Define the RNN Structure
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])




#Fit on the training data.
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=100,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0000001)])

#Process the test data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#Evaluate the test set
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
