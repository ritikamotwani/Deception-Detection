import os, string, glob
import pandas as pd
import numpy as np
from random import shuffle
from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from statistics import mean
from utilities import *

#--------------------------------opinion Spam Data----------------------------------------------------------------------------------------------

sources = [[] for i in range(4)]

datapath='./Spam_Detection_Data/'
sources = [[] for i in range(4)]

read_fileNames(sources[0], datapath,'deceptive_neg')
read_fileNames(sources[1], datapath,'truthful_neg')
read_fileNames(sources[2], datapath,'deceptive_pos')
read_fileNames(sources[3], datapath,'truthful_pos')

sort_Lists(sources,len(sources))

df=[]

read_sort_CSV(df, datapath, 'LIWC_Negative_Deceptive.csv','Filename')
read_sort_CSV(df, datapath, 'LIWC_Negative_Truthful.csv','Filename')
read_sort_CSV(df, datapath, 'LIWC_Positive_Deceptive.csv','Filename')
read_sort_CSV(df, datapath, 'LIWC_Positive_Truthful.csv','Filename')


dfLIWC=pd.concat((df[0],df[1],df[2],df[3])).iloc[:,2:]

labels=np.empty((len(sources)*len(sources[0])),dtype=int)
np.concatenate((np.ones((400),dtype=int),np.zeros((400),dtype=int),np.ones((400),dtype=int),np.zeros((400),dtype=int)),out=labels)

#--------------------------------Real-Life Data----------------------------------------------------------------------------------------------

# #create source
# sources = []
# datapath = './Real_Life_Trial_Data/'

# #read file names from datapath
# read_fileNames(sources, datapath,'Deceptive')
# deceptiveCount=len(sources)
# read_fileNames(sources, datapath,'Truthful')
# truthCount=len(sources)-deceptiveCount

# #read LIWC output CSV files
# df=[]
# read_sort_CSV(df, datapath, 'LIWC2015_Deceptive.csv','Filename')
# read_sort_CSV(df, datapath, 'LIWC2015_Truthful.csv','Filename')

# dfLIWC=pd.concat((df[0],df[1])).iloc[:,2:]

# #create label array corresponding to text files
# labels=np.empty(len(sources))
# np.concatenate((np.ones(deceptiveCount, dtype=int),np.zeros(truthCount, dtype=int)),out=labels)

#--------------------------------------------------------------------------------------------------------------------------------------------

#read text files from source list
text=[]
readFilesFromSources(text,sources)

#read stopwords file
with open('./Spam_Detection_Data/stopwords.txt') as f_stop:
        stopwords=f_stop.read().splitlines()

#encode Deceptive/Truthful Class labels
encoder = preprocessing.LabelEncoder()
labels=encoder.fit_transform(labels)

#train-test split
train_txt, valid_txt, train_LIWC, valid_LIWC, train_labels, valid_labels= model_selection.train_test_split(text, dfLIWC, labels, test_size = 0.10, random_state = 0)

i=1692
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
print("Accuracy score = %.3f\nF1 score = %.3f"%(result['accuracy'],result['f1']))