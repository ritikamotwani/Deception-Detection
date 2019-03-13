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

#Defining Variables
sources = []
labels = []
text = []

# Reading and storing dataset
def read_file():
    trainingPath = './Spam_Detection_Data/'
    for home, dirs, files in os.walk(trainingPath+'deceptive_neg'):
        for filename in files:
            sources.append(home+'/'+filename)
            labels.append(1)

    for home, dirs, files in os.walk(trainingPath+'deceptive_pos'):
        for filename in files:
            sources.append(home+'/'+filename)
            labels.append(1)

    for home, dirs, files in os.walk(trainingPath+'truthful_neg'):
        for filename in files:
            sources.append(home+'/'+filename)
            labels.append(0)

    for home, dirs, files in os.walk(trainingPath+'truthful_pos'):
        for filename in files:
            sources.append(home + '/' + filename)
            labels.append(0)

    for source in sources:
        with open(source) as f_input:
            text.append(f_input.read())

def create_train_test_set():
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = text
    trainDF['label'] = labels

    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size = 0.10, random_state = 0, shuffle=False)

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), lowercase=True)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return accuracy_score(predictions, valid_y)

# SVM on Ngram Level TF IDF Vectors
read_file()
xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y = create_train_test_set()
accuracy = train_model(svm.SVC(kernel='linear'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print("SVM, N-Gram Vectors: ", accuracy)
