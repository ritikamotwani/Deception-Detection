import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utilities import train_model
from sklearn.ensemble import RandomForestClassifier

#Read LIWC output file

df = pd.read_csv("./Real_Life_Trial_Data/LIWC_RealLife.csv")

#split training and testing data
x_train, x_test, y_train, y_test = model_selection.train_test_split(df.iloc[:,2:-1],df.iloc[:,-1], test_size=0.15, random_state=0)

# Define classifiers
classifier1=GaussianNB()
classifier2=SVC(kernel='linear')
classifier3 = RandomForestClassifier(n_estimators=2, random_state=0, max_features='auto', min_samples_split=2)

#K-Fold cross validation on training set
k=5
kf=KFold(n_splits=k,shuffle=True,random_state=0)
print("K-Fold cross validation (K=%d)"%k)
i=1
for train_index, valid_index in kf.split(x_train):
    print("\nFold ",i)
    i+=1
    training_data,valid_data=x_train.iloc[train_index],x_train.iloc[valid_index]
    expected_labels = y_train.iloc[valid_index]

    result1=train_model(classifier1,training_data,y_train.iloc[train_index], valid_data, expected_labels)
    print("NB result : ",result1)

    result2=train_model(classifier2,training_data,y_train.iloc[train_index], valid_data, expected_labels)   
    print("SVM result : ",result2)

    result3=train_model(classifier3, training_data, y_train.iloc[train_index], valid_data, expected_labels)   
    print("Random Forest result : ",result3)


#Final classification
print("Train-test classification...\n")
result1=train_model(classifier1,x_train, y_train, x_test, y_test)
print("NB result : ",result1)
result2=train_model(classifier2,x_train, y_train, x_test, y_test)
print("SVM result : ",result2)
result3=train_model(classifier3,x_train, y_train, x_test, y_test)
print("Random Forest result : ",result3)