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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from utilities import read_fileNames, readFilesFromSources, train_model

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
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size = 0.10, random_state = 0)

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), lowercase=True)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# SVM on Ngram Level TF IDF Vectors
result = train_model(svm.SVC(kernel='linear'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print("Accuracy score = %.3f\nF1 score = %.3f"%(result['accuracy'],result['f1']))

