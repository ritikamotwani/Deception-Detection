import os, string, glob
import pandas as pd
import numpy as np
import itertools
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from statistics import mean

sources = [[] for i in range(4)]

# for infile in glob.glob(datapath+'deceptive_neg/*.txt'):

datapath='./Spam_Detection_Data/'
for home, dirs, files in os.walk(datapath+'deceptive_neg'):
    for filename in files:
        sources[0].append(home+'/'+filename)

sources[0]=sorted(sources[0])

for home, dirs, files in os.walk(datapath+'truthful_neg'):
    for filename in files:
        sources[1].append(home+'/'+filename)

sources[1]=sorted(sources[1])

for home, dirs, files in os.walk(datapath+'deceptive_pos'):
    for filename in files:
        sources[2].append(home+'/'+filename)

sources[2]=sorted(sources[2])

for home, dirs, files in os.walk(datapath+'truthful_pos'):
    for filename in files:
        sources[3].append(home+'/'+filename)

sources[3]=sorted(sources[3])

df=[]
df.append(pd.read_csv(datapath+'LIWC_Negative_Deceptive.csv'))
df[0] = df[0].sort_values(by=['Filename'])

df.append(pd.read_csv(datapath+'LIWC_Negative_Truthful.csv'))
df[1] = df[1].sort_values(by=['Filename'])

df.append(pd.read_csv(datapath+'LIWC_Positive_Deceptive.csv'))
df[2] = df[2].sort_values(by=['Filename'])

df.append(pd.read_csv(datapath+'LIWC_Positive_Truthful.csv'))
df[3] = df[3].sort_values(by=['Filename'])

dfLIWC=pd.concat((df[0],df[1],df[2],df[3])).iloc[:,2:]

labels=np.empty((len(sources)*len(sources[0])),dtype=int)
np.concatenate((np.ones((400),dtype=int),np.zeros((400),dtype=int),np.ones((400),dtype=int),np.zeros((400),dtype=int)),out=labels)

text=[]
for source in list(itertools.chain.from_iterable(sources)):
        with open(source) as f_input:
                text.append(f_input.read())


# print(np.concatenate((np.column_stack((sources,labels)),X),axis=1))

with open('./stopwords.txt') as f_stop:
        stopwords=f_stop.read().splitlines()

encoder = preprocessing.LabelEncoder()
labels=encoder.fit_transform(labels)

#train-test split
train_txt, valid_txt, train_LIWC, valid_LIWC, train_labels, valid_labels= model_selection.train_test_split(text, dfLIWC, labels, test_size = 0.10, random_state = 0)

# for i in range(1690,1695,1):
i=1692
print("Max features = %d\n"%i)
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), lowercase=True, stop_words=stopwords, max_features=i)
tfidf_vect_ngram.fit(train_txt)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_txt)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_txt)

train_X=np.empty((len(train_txt),i+93))
# print(xtrain_tfidf_ngram.todense().shape)
np.concatenate((xtrain_tfidf_ngram.todense(),train_LIWC),axis=1,out=train_X)

valid_X=np.empty((len(valid_txt),i+93))
np.concatenate((xvalid_tfidf_ngram.todense(),valid_LIWC),axis=1,out=valid_X)

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

def train_model(classifier, feature_vector_train, train_label, feature_vector_valid, valid_label):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, train_label)
        
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        return {'accuracy':accuracy_score(predictions, valid_label),'f1':f1_score(predictions, valid_label)}

# SVM on Ngram Level TF IDF Vectors
print("Classifying test data using SVM...")
result = train_model(clf, train_X, train_labels, valid_X, valid_labels)
print("Accuracy score = %.3f\nF1 score = %.3f"%(result['accuracy'],result['f1']))