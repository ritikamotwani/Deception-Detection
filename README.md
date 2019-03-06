# Deception-Detection

Classifiers added to detect deception through transcripts.
<br />Dataset Used: 
1. Real life trial data collected during a series of experiments at Michigan (http://web.eecs.umich.edu/~zmohamed/PDFs/Trial.ICMI.pdf) (Folder - dataset)
2. Deceptive Opinion Spam Corpus v1.4 (https://myleott.com/op-spam.html)

## Installation
</br>
Run the following command to install all python packages that'll be used in the project:
```
pip install -r requirements.txt
```
</br>
## Approaches
</br>
1. The first basic approach: The file SVM_realLife.py classifies using PCA, then SVM. 
```
   To execute:
      python3 SVM.py
```

2. Using SVM with ngrams approach(bigrams and trigrams) with tf-idf on real lige trial dataset
```
   To execute:
      python3 SVM_nGrams_realLife.py
```

3.  Using SVM with ngrams approach(bigrams and trigrams) with tf-idf and removing the stop words on real life trial dataset
```
   To execute:
      python3 SVM_nGrams_removingStopWords_realLife.py
```

4. SVM with the parameters - tf-idf and bigrams on spam detection dataset
```
   To execute:
      python3 SVM_nGrams_spamDetection.py
```

5. Applying RNN on real life dataset
```
   To execute:
      python3 RNN_on_realLife.py
```

6. Applying RNN on spam detection dataset
```
   To execute:
      python3 RNN_spamDetection.py
```

7. Classification done using Random Forest Algorithm. LIWC is initially applied to the spam detection dataset and then classification is done.
```
   To execute:
      python3 randomForest_LIWC_spamDetection.py
```
