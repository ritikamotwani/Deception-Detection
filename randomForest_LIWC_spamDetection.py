import numpy as np
import os
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from input import *

#--------------------------------Opinion Spam Data----------------------------------------------------------------------------------------------

text,labels=readTxt_Spam()
i=3200
#--------------------------------Real-Life Data----------------------------------------------------------------------------------------------

# text,labels=readTxt_RealLife()
# dfLIWC=readLIWC_RealLife()
# i=2000
#--------------------------------------------------------------------------------------------------------------------------------------------
#read stopwords file
with open('./stopwords.txt') as f_stop:
        stopwords=f_stop.read().splitlines()

#train-test split
train_txt, valid_txt, train_labels, valid_labels= model_selection.train_test_split(text, labels, test_size = 0.10, random_state = 0)


print("Max features = %d\n"%i)
#extract N-gram features from text
xtrain_tfidf_ngram, xvalid_tfidf_ngram = ngram_transform(train_txt, valid_txt, 2, stopwords,i)

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(xtrain_tfidf_ngram, train_labels)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(xtrain_tfidf_ngram, train_labels)

predictions = clf.predict(xvalid_tfidf_ngram)
print(accuracy_score(valid_labels, predictions))
