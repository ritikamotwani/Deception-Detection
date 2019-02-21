# gensim modules
import gensim
from gensim import utils
from gensim import corpora,models
# numpy
import numpy
# classifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from random import shuffle

#create source
sources = []
labels = []

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
