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
from sklearn.ensemble import RandomForestClassifier
from utilities import read_fileNames, readFilesFromSources, train_model, ngram_transform

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
def create_train_test_set():

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(text, labels, test_size = 0.10, random_state = 0, shuffle=True)

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # ngram level tf-idf
    xtrain_tfidf_ngram, xvalid_tfidf_ngram = ngram_transform(train_x, valid_x, n=2)
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y


# SVM on Ngram Level TF IDF Vectors
xtrain_tfidf_ngram, xvalid_tfidf_ngram, train_y, valid_y = create_train_test_set()
accuracy_SVM = train_model(svm.SVC(kernel='linear'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
accuracy_RF = train_model(RandomForestClassifier(n_estimators=2, random_state=0, max_features='auto', min_samples_split=2), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
accuracy_NB = train_model(naive_bayes.MultinomialNB(alpha=0, class_prior=None, fit_prior=False), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print('\n')
print('The statistics for the classifiers SVM, Naïve Bayes, Random Forest are: ')
print("1. SVM, N-Gram Vectors: ", accuracy_SVM)
print("2. Random Forest, N-Gram Vectors: ", accuracy_RF)
print("3. Naive Bayes, N-Gram Vectors: ", accuracy_NB)
