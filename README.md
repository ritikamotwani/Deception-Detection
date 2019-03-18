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

<b><u>Folder: OpinionSpam</u></b>

1. Parameters: NGram Approach 
Classifiers: SVM, NB, Random Forest 
```
   To execute:
      python3 Ngrams_And_Classifiers.py 
```
SVM
<br/>
![Accuracy: 91.25%](https://img.shields.io/badge/Accuracy-91.25%25-blue.svg)
<br/>
![F1Score: 90.27%](https://img.shields.io/badge/F1Score-90.27%25-blue.svg)
<br/>
NB
<br/>
![Accuracy: 85.6%](https://img.shields.io/badge/Accuracy-85.6%25-blue.svg)
<br/>
![F1Score: 83.6%](https://img.shields.io/badge/F1Score-83.6%25-blue.svg)
<br/>
Random Forest
<br/>
![Accuracy: 68.7%](https://img.shields.io/badge/Accuracy-68.7%25-blue.svg)
<br/>
![F1Score: 62.1%](https://img.shields.io/badge/F1Score-62.1%25-blue.svg)
<br/>

2. Parameters: LIWC
   Classifiers: SVM, NB, Random Forest
```
   To execute:
      python3 LIWC_And_Classifiers.py 
```
NB
<br/>
![Accuracy: 65.4%](https://img.shields.io/badge/Accuracy-65.4%25-blue.svg)
<br/>
![F1Score: 71%](https://img.shields.io/badge/F1Score-71%25-blue.svg)
<br/>
SVM
<br/>
![Accuracy: 79.1%](https://img.shields.io/badge/Accuracy-79.1%25-blue.svg)
<br/>
![F1Score: 79.8%](https://img.shields.io/badge/F1Score-79.8%25-blue.svg)
<br/>
Random Forest
<br/>
![Accuracy: 67.9%](https://img.shields.io/badge/Accuracy-67.9%25-blue.svg)
<br/>
![F1Score: 59.6%](https://img.shields.io/badge/F1Score-59.6%25-blue.svg)

<br/>

3.  Parameters: NGrams, LIWC
    Classifiers: SVM
```
   To execute:
      python3 SVM_Ngrams_LIWC.py
```
![Accuracy: 84.4%](https://img.shields.io/badge/Accuracy-84.4%25-blue.svg)
<br/>
![F1Score: 83.2%](https://img.shields.io/badge/F1Score-83.2%25-blue.svg)
<br/>

4. Recurrent Neural Networks
```
   To execute:
      python3 RNN
```
![Accuracy: 71%](https://img.shields.io/badge/Accuracy-71%25-blue.svg)
<br/>

### Real Life Dataset
<b><u>Folder: RealLife</u></b>

1. Classifiers: SVM, NB
```
   To execute:
      python3  Classifiers.py
```
SVM
<br/>
![Accuracy: 73.7%](https://img.shields.io/badge/Accuracy-73.7%25-blue.svg)
<br/>
![F1Score: 78.3%](https://img.shields.io/badge/F1Score-78.3%25-blue.svg)
<br/>
NB
<br/>
![Accuracy: 68.4%](https://img.shields.io/badge/Accuracy-68.4%25-blue.svg)
<br/>
![F1Score: 66.7%](https://img.shields.io/badge/F1Score-66.7%25-blue.svg)
<br/>

2. Parameters: NGram Approach 
Classifiers: SVM, NB, Random Forest 
```
   To execute:
      python3 Ngrams_And_Classifiers.py 
```
SVM
<br/>
![Accuracy: 76.9%](https://img.shields.io/badge/Accuracy-76.9%25-blue.svg)
<br/>
![F1Score: 80%](https://img.shields.io/badge/F1Score-80%25-blue.svg)
<br/>
NB
<br/>
![Accuracy: 69.2%](https://img.shields.io/badge/Accuracy-69.2%25-blue.svg)
<br/>
![F1Score: 71.4%](https://img.shields.io/badge/F1Score-71.4%25-blue.svg)
<br/>
Random Forest
<br/>
![Accuracy: 46.1%](https://img.shields.io/badge/Accuracy-46.1%25-blue.svg)
<br/>
![F1Score: 53.3%](https://img.shields.io/badge/F1Score-53.3%25-blue.svg)
<br/>

3.  Parameters: LIWC
   Classifiers: SVM, NB, Random Forest
```
   To execute:
      python3 LIWC_And_Classifiers.py 
```
NB
<br/>
![Accuracy: 63.1%](https://img.shields.io/badge/Accuracy-63.1%25-blue.svg)
<br/>
![F1Score: 70.99%](https://img.shields.io/badge/F1Score-70.99%25-blue.svg)
<br/>
SVM
<br/>
![Accuracy: 57.8%](https://img.shields.io/badge/Accuracy-57.8%25-blue.svg)
<br/>
![F1Score: 60%](https://img.shields.io/badge/F1Score-60%25-blue.svg)
<br/>
Random Forest
<br/>
![Accuracy: 52.6%](https://img.shields.io/badge/Accuracy-52.6%25-blue.svg)
<br/>
![F1Score: 40%](https://img.shields.io/badge/F1Score-40%25-blue.svg)

<br/>

4. RNN
```
   To execute:
      python3 RNN.py
```
![Accuracy: 57.9%](https://img.shields.io/badge/Accuracy-57.9%25-blue.svg)
<br/>