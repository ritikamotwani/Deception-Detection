import os, string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn import model_selection, preprocessing, naive_bayes
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from utilities import read_fileNames, readFilesFromSources, train_model, ngram_transform, sort_Lists, stemText
from input import readTxt_Spam, readTxt_RealLife

#--------------------------------Opinion Spam Data-------------------------------------------------------------------------------------------

text,labels=readTxt_Spam()
i=3200

#--------------------------------Real-Life Data----------------------------------------------------------------------------------------------

# text,labels=readTxt_RealLife()
# i=3000

#--------------------------------------------------------------------------------------------------------------------------------------------

#read stopwords file
with open('./stopwords.txt') as f_stop:
    stopwords=f_stop.read().splitlines()

# stemming of dataset
# text=stemText(text)

#create train-test split
train_txt, valid_txt, train_labels, valid_labels= model_selection.train_test_split(text, labels, test_size = 0.10, random_state = 0)

# max_acc=0
# max_f=0
print("\nMax features = %d"%i)
#extract N-gram features from text
xtrain_tfidf_ngram, xvalid_tfidf_ngram = ngram_transform(train_txt, valid_txt, 2, stopwords=stopwords, max_Features=i)

accuracy_SVM = train_model(svm.SVC(kernel='linear'), xtrain_tfidf_ngram, train_labels, xvalid_tfidf_ngram, valid_labels)
# accuracy_RF = train_model(RandomForestClassifier(n_estimators=2, random_state=0, max_features='auto', min_samples_split=2), xtrain_tfidf_ngram, train_labels, xvalid_tfidf_ngram, valid_labels)
accuracy_NB = train_model(naive_bayes.MultinomialNB(alpha=0, class_prior=None, fit_prior=False), xtrain_tfidf_ngram, train_labels, xvalid_tfidf_ngram, valid_labels)

print('The statistics for the classifiers SVM, Naïve Bayes, Random Forest are: ')
print("1. SVM, N-Gram Vectors: ", accuracy_SVM)
# print("2. Random Forest, N-Gram Vectors: ", accuracy_RF)
print("3. Naive Bayes, N-Gram Vectors: ", accuracy_NB)

#     if(accuracy_SVM['accuracy']>max_acc):
#         max_acc=accuracy_SVM['accuracy']
#         max_f=i

# print("Max Acc = %.4f\n Max features = %d"%(max_acc,max_f))