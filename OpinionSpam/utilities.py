import os, string
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

def read_fileNames(src, datapath, subfolder=None):
    for home, dirs, files in os.walk(datapath+subfolder):
        for filename in files:
            src.append(home+'/'+filename)

def sort_Lists(sources, length):
    for i in range(0,length):
        sources[i]=sorted(sources[i])

def read_sort_CSV(df, datapath, filename, sort_column):
    df.append(pd.read_csv(datapath+filename))
    index=len(df)-1
    df[index]=df[index].sort_values(by=sort_column)

def readFilesFromSources(text, sources):
    if np.asarray(sources).ndim>1:
        sources=list(itertools.chain.from_iterable(sources))
    for source in sources:
        with open(source) as f_input:
            text.append(f_input.read())

def ngram_transform(train_txt, valid_txt, n, stopwords=None, max_Features=None):
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,n), lowercase=True, stop_words=stopwords, max_features=max_Features)
    tfidf_vect_ngram.fit(train_txt)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_txt)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_txt)
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram

def train_model(classifier, feature_vector_train, train_label, feature_vector_valid, valid_label):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, train_label)
        
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    return {'accuracy':accuracy_score(predictions, valid_label),'f1':f1_score(predictions, valid_label)}
