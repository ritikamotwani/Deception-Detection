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

### Opinion Spam Dataset:
Folder: OpinionSpam

1. Parameters: NGram Approach
   Classifiers: SVM, NB, Random Forest 
```
   To execute:
      python3 Ngrams_And_Classifiers.py 
```
SVM
![Accuracy: 91.25%](https://img.shields.io/badge/Accuracy-91.25%25-blue.svg)
![F1Score: 90.27%](https://img.shields.io/badge/F1Score-90.27%25-blue.svg)
NB
![Accuracy: 85.6%](https://img.shields.io/badge/Accuracy-85.6%25-blue.svg)
![F1Score: 83.6%](https://img.shields.io/badge/F1Score-83.6%25-blue.svg)
Random Forest
![Accuracy: 68.7%](https://img.shields.io/badge/Accuracy-68.7%25-blue.svg)
![F1Score: 62.1%](https://img.shields.io/badge/F1Score-62.1%25-blue.svg)
<br/>

2. Parameters: LIWC
   Classifiers: SVM, NB, Random Forest
```
   To execute:
      python3 LIWC_And_Classifiers.py 
```
NB
![Accuracy: 65.4%](https://img.shields.io/badge/Accuracy-65.4%25-blue.svg)
![F1Score: 71%](https://img.shields.io/badge/F1Score-71%25-blue.svg)
SVM
![Accuracy: 79.1%](https://img.shields.io/badge/Accuracy-79.1%25-blue.svg)
![F1Score: 79.8%](https://img.shields.io/badge/F1Score-79.8%25-blue.svg)
Random Forest
![Accuracy: 67.9%](https://img.shields.io/badge/Accuracy-67.9%25-blue.svg)
![F1Score: 59.6%](https://img.shields.io/badge/F1Score-59.6%25-blue.svg)

<br/>

3.  Parameters: NGrams, LIWC
    Classifiers: SVM
```
   To execute:
      python3 SVM_Ngrams_LIWC.py
```
![Accuracy: 84.4%](https://img.shields.io/badge/Accuracy-84.4%25-blue.svg)
![F1Score: 83.2%](https://img.shields.io/badge/F1Score-83.2%25-blue.svg)
<br/>

4. Recurrent Neural Networks
```
   To execute:
      python3 RNN
```
![Accuracy: 71%](https://img.shields.io/badge/Accuracy-71%25-blue.svg)
<br/>