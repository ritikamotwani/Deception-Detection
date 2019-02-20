# gensim modules
import gensim
from gensim import utils
from gensim import corpora,models
# numpy
import numpy
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import os
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

PCA_Applied = True
PCA_nComponents = 50
stoplist = set('for a of the and to in'.split())

class Texts(object):
    def __init__(self, sources):
        self.sources = sources


    def to_vector(self):
        self.sentences = []
        for source in self.sources:
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    words = utils.to_unicode(line).split()
                    words = [word for word in words if word not in stoplist]
                    self.sentences.append(words)
        return self.sentences


#create source
sources = []
labels = []
sources_test = []

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
        sources_test.append(home + '/' + filename)
        labels.append(1)

for home, dirs, files in os.walk(testPath + 'Truthful'):
    for filename in files:
        sources_test.append(home + '/' + filename)
        labels.append(0)

corpus = Texts(sources).to_vector()
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(text) for text in corpus]
model = models.TfidfModel(corpus)
corpus = [text for text in model[corpus]]
print(len(corpus))
if len(corpus):
    text_matrix = gensim.matutils.corpus2dense(corpus,num_terms = len(dictionary.token2id)).Texts
    if PCA_Applied:
        pca = PCA(n_components=PCA_nComponents)
        text_matrix = pca.fit_transform(text_matrix)
    ppl = Pipeline([
              ('ngram', CountVectorizer(ngram_range=(1, 2), analyzer='char')),
              ('clf',   SVC())
      ])
    model = ppl.fit(sources, labels[:100])
    pred_labels = model.predict(sources_test)
    print('SVM:')
    print('Predictions:')
    print(classification_report(labels[100:], pred_labels))
