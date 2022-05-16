# CSCI-636
 Big Data Management & Analytics 

## COURSE DESCRIPTION 
 Organizations today are generating massive amounts of data that are too large and unstructured to fit in relational databases. Organizations and enterprises are turning to massively parallel computing solutions such as Hadoop. The  Apache  Hadoop  platform  allows  for  distributed  processing  of  large  data  sets  across  clusters  of  computers using the map and reduce programming model.  Students will gain an in-depth understanding of how MapReduce and Distributed File Systems work. In addition, they will be able to author Hadoop-based MapReduce applications in Java and use Hadoop subprojects Hive and Pig to build powerful data processing applications. 

## Amazon Reviews Sentiment Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w5zW_3l02mLZxcDmzQCpdXd5cSeNRS8B?usp=sharing)
## Problem:
 There are millions of production review on Amazon. Manually analyzing Amazon reviews is extremely time-consuming and inefficient. And it would be an impossible task to find trends or patterns over time.

## Goal:
 Doing a Sentiment Analysis on the customer reviews with help of a machine learning model and then using Big Data to process large volumes of reviews.

## Data Source: 
 [source](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
 [source2](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)


## procedure

### Step 1: load packages
 
 ```python
 import pandas as pd
 import matplotlib.pyplot as plt
 import sklearn
 from sklearn.feature_selection import VarianceThreshold

 import nltk
 nltk.download('stopwords')
 ```

### Step 2: Preprocess the data

### Step 3: Fitting simple Classfier model to the training set
 Training in different classifier
 to find the best fitting classifier for this dataset

 * [Naive Bayes Classifier](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf) 
 * [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html#)
 * [K Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#)
 * Gradient Boosting Classifier

### Step 4: Predicting the Result

### Step 5: Visulization 

## Citation:
 * [Learning to Classify Text](https://www.nltk.org/book/ch06.html)
 * [NLP with Python](https://rpubs.com/pjozefek/669929)
 * [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
 * [RBF SVM parameters](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
 * [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
 * Géron, Aurélien, and Rebecca Demarest. “3. Classification/ Performance Measures/ Precision and Recall.” Hands-on Machine Learning with Scikit-Learn and Tensorflow: Concepts, Tools, and Techniques to Build Intelligent Systems, O'Reilly, 2019.

