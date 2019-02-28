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


#create source
sources = []
labels = []
stoplist = set('for a of the and to in'.split())
stopwords = ['for', 'a', 'of', 'the', 'and', 'to', 'in']

trainingPath = './dataset/TrainingSet/'
for home, dirs, files in os.walk(trainingPath+'Deceptive'):
    for filename in files:
        sources.append(home+'/'+filename)
        labels.append(1)

for home, dirs, files in os.walk(trainingPath+'Truthful'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(0)

testPath = './dataset/TestingSet/'
for home, dirs, files in os.walk(testPath + 'Deceptive'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(1)

for home, dirs, files in os.walk(testPath + 'Truthful'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(0)
text = []
for source in sources:
    with open(source) as f_input:
        text.append(f_input.read())
for i in range(121):
    querywords = text[i].split()
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    text[i] = ' '.join(resultwords)
    print(text[i])
    print('\n')
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = text
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], random_state = np.random.seed(0), test_size=0.15)

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

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,1), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(7,8), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return accuracy_score(predictions, valid_y)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)