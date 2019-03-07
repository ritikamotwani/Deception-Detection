import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

labels = []

trainingPath = './dataset1/'
for home, dirs, files in os.walk(trainingPath+'deceptive1'):
    for filename in files:
        labels.append(1)

for home, dirs, files in os.walk(trainingPath+'deceptive2'):
    for filename in files:
        labels.append(1)

for home, dirs, files in os.walk(trainingPath+'truthful1'):
    for filename in files:
        labels.append(0)

for home, dirs, files in os.walk(trainingPath+'truthful2'):
    for filename in files:
        labels.append(0)


data1 = pd.read_csv('deceptive1.csv', index_col=0)
data2 = pd.read_csv('deceptive2.csv', index_col=0)
data3 = pd.read_csv('truthful1.csv', index_col=0)
data4 = pd.read_csv('truthful2.csv', index_col=0)
frames = [data1, data2, data3, data4]
data = pd.concat(frames)

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=num_test, random_state=0)

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
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)



predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
