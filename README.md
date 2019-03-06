# Deception-Detection

Classifiers added to detect deception through transcripts.
<br />Dataset Used: 
1. Real life trial data collected during a series of experiments at Michigan (http://web.eecs.umich.edu/~zmohamed/PDFs/Trial.ICMI.pdf) (Folder - dataset)
2. Deceptive Opinion Spam Corpus v1.4 (https://myleott.com/op-spam.html)

## Installation

Run the following command to install all python packages that'll be used in the project:
```
pip install -r requirements.txt
```

## Approaches

1. The first basic approach: The file SVM_realLife.py classifies using PCA, then SVM. 
```
   To execute:
      python3 SVM.py
```
![Accuracy: 67%](https://img.shields.io/badge/Accuracy-67%25-blue.svg)
<br/>

2. Using SVM with ngrams approach(bigrams and trigrams) with tf-idf on real lige trial dataset
```
   To execute:
      python3 SVM_nGrams_realLife.py
```
![Accuracy: 69.23%](https://img.shields.io/badge/Accuracy-69.23%25-blue.svg)
<br/>

3.  Using SVM with ngrams approach(bigrams and trigrams) with tf-idf and removing the stop words on real life trial dataset
```
   To execute:
      python3 SVM_nGrams_removingStopWords_realLife.py
```
![Accuracy: 68.4%](https://img.shields.io/badge/Accuracy-68.4%25-blue.svg)
<br/>

4. SVM with the parameters - tf-idf and bigrams on spam detection dataset
```
   To execute:
      python3 SVM_nGrams_spamDetection.py
```
![Accuracy: 86.8%](https://img.shields.io/badge/Accuracy-86.8%25-blue.svg)
<br/>

5. Applying RNN on real life dataset
```
   To execute:
      python3 RNN_on_realLife.py
```
![Accuracy: 57.9%](https://img.shields.io/badge/Accuracy-57.9%25-blue.svg)
<br/>

6. Applying RNN on spam detection dataset
```
   To execute:
      python3 RNN_spamDetection.py
```
![Accuracy: 76.2%](https://img.shields.io/badge/Accuracy-76.2%25-blue.svg)
<br/>

7. Classification done using Random Forest Algorithm. LIWC is initially applied to the spam detection dataset and then classification is done.
```
   To execute:
      python3 randomForest_LIWC_spamDetection.py
```
![Accuracy: 75.9%](https://img.shields.io/badge/Accuracy-75.9%25-blue.svg)
<br/>
