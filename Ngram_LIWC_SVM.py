import os, string, glob
import pandas as pd
import numpy as np
from random import shuffle
from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from statistics import mean
from input import *
from utilities import *

#--------------------------------Opinion Spam Data----------------------------------------------------------------------------------------------

text,labels=readTxt_Spam()
dfLIWC=readLIWC_Spam()
i=1600
#--------------------------------Real-Life Data----------------------------------------------------------------------------------------------

# text,labels=readTxt_RealLife()
# dfLIWC=readLIWC_RealLife()
# i=2000
#--------------------------------------------------------------------------------------------------------------------------------------------

#read stopwords file
with open('./stopwords.txt') as f_stop:
        stopwords=f_stop.read().splitlines()

# stemming of dataset
# text=stemText(text)

#normalize LIWC input
dfLIWC=normalize(dfLIWC,norm='l2')

#train-test split
train_txt, valid_txt, train_LIWC, valid_LIWC, train_labels, valid_labels= model_selection.train_test_split(text, dfLIWC, labels, test_size = 0.10, random_state = 0)


print("Max features = %d\n"%i)
#extract N-gram features from text
xtrain_tfidf_ngram, xvalid_tfidf_ngram = ngram_transform(train_txt, valid_txt, 2, stopwords,i)

#concatenate N-gram features with LIWC features
train_X=np.concatenate((xtrain_tfidf_ngram.todense(),train_LIWC),axis=1)
valid_X=np.concatenate((xvalid_tfidf_ngram.todense(),valid_LIWC),axis=1)

clf=svm.SVC(kernel='linear')

# K-fold cross validation to test model and data variance
# k=10
# print("Performing %d fold cross validation..."%k)
# score = cross_validate(clf, train_X, train_labels, cv=k, scoring=['accuracy','f1_macro'])
# # print("SVM Accuracy = %.3f +- %.3f"%(score2.mean(),score2.std()*2))
# print("Results:\nAccuracy:",score['test_accuracy'],"\nF1 score:",score['test_f1_macro'])
# avgF1=mean(score['test_f1_macro'])
# avgAcc=mean(score['test_accuracy'])
# print("Overall F1 = %.3f\nAccuracy = %.3f\n"%(avgF1,avgAcc))

# Final classification
print("Classifying test data using SVM...")
result = train_model(clf, train_X, train_labels, valid_X, valid_labels)
print("Accuracy score = %.4f\nF1 score = %.4f"%(result['accuracy'],result['f1']))