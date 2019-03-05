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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 

#create source
sources = []
labels = []

trainingPath = './dataset1/'
for home, dirs, files in os.walk(trainingPath+'deceptive1'):
    for filename in files:
        sources.append(home+'/'+filename)
        labels.append(1)

for home, dirs, files in os.walk(trainingPath+'deceptive2'):
    for filename in files:
        sources.append(home+'/'+filename)
        labels.append(1)

for home, dirs, files in os.walk(trainingPath+'truthful1'):
    for filename in files:
        sources.append(home+'/'+filename)
        labels.append(0)

for home, dirs, files in os.walk(trainingPath+'truthful2'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(0)

text = []
for source in sources:
    with open(source) as f_input:
        text.append(f_input.read())
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = text
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size = 0.1, random_state = 0)

kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 

for train_index, test_index in kf.split(X):
      print("Train:", train_index, "Validation:",test_index)
      train_x, valid_x = X[train_index], X[test_index] 
      y_train, y_test = y[train_index], y[test_index]

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=24)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return accuracy_score(predictions, valid_y)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(degree = 5, random_state = 0, C=10), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
accuracy = cross_val_score(accuracy, xtrain_tfidf_ngram, train_y, cv=5)
print("SVM, N-Gram Vectors: ", accuracy)