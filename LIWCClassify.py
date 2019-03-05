import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
import random
import operator


# df = pd.read_csv("./LIWC_output_data/LIWC_All.csv")
df = pd.read_csv("./LIWC_output_data/LIWC_RealLife.csv")
X=df.iloc[:,1:]
# print(X)
kf=KFold(n_splits=5,shuffle=True,random_state=0)
netAccuracy1=0
netAccuracy2=0
classifier1=GaussianNB()
classifier2=SVC(kernel='linear')
i=1
for train_index, test_index in kf.split(X):
    print("Fold ",i)
    i+=1
    training_data,test_data=X.iloc[train_index],X.iloc[test_index]
    expected_labels = test_data.iloc[:,-1]

    classifier1.fit(training_data.iloc[:,:-1], training_data.iloc[:,-1])
    classifier2.fit(training_data.iloc[:,:-1], training_data.iloc[:,-1])

    accuracy1 = classifier1.score(test_data.iloc[:,:-1], expected_labels)
    print("NB accuracy = %.3f"%accuracy1)
    accuracy2 = classifier2.score(test_data.iloc[:,:-1], expected_labels)
    print("SVM accuracy = %.3f"%accuracy2)
    netAccuracy1+=accuracy1
    netAccuracy2+=accuracy2
print("Overall accuracy:\nNB = %.3f\nSVM = %.3f"%(netAccuracy1/5,netAccuracy2/5))