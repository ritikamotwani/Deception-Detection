import os,string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, preprocessing, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix
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
trainDF = pd.DataFrame()
trainDF['text'] = text
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.15, random_state=0)

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
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x) 

# Naive Bayes on Word Level TF IDF Vectors
result = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("NB, WordLevel TF-IDF: Accuracy=%.3f\tF1=%.3f"%(result['accuracy'],result['f1']))
result1 = train_model(svm.SVC(kernel="linear"), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("SVM, WordLevel TF-IDF: Accuracy=%.3f\tF1=%.3f"%(result1['accuracy'],result1['f1']))
