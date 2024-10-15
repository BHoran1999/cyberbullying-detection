# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:23:11 2022

@author: bhbri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import codecs

from sklearn import neighbors
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

import sys




df = pd.read_csv("cyberbully_balance.csv")

tweets_text = df['Comment']

def cleaning(tweets_text):
    
    tweets_text2 = tweets_text
    
        #Remove URLS
    tweets_text2 = tweets_text2.replace(r'\w+:\/\/\S+', ' ', regex=True)
    
    tweets_text2 = tweets_text2.replace(r'https', ' ', regex=True)
    
        #Remove joining of words as soon as new line is made
    tweets_text2 = tweets_text2.replace(r'\n', ' ', regex=True)
    tweets_text2 = tweets_text2.replace(r'\\n', ' ', regex=True)
    tweets_text2 = tweets_text2.replace(r'_', ' ', regex=True)
    tweets_text2 = tweets_text2.replace(r'-', ' ', regex=True)
    
        #Remove Twitter Usernames
    tweets_text2 = tweets_text2.replace(r'(\A|\s)@(\w+)+[a-zA-Z0-9_\.]', ' ', regex=True)
    
        #Remove non-alphabets
    tweets_text2 = tweets_text2.replace(r'[^a-zA-Z ]\s?','',regex=True)
    
        #Whitespace Formatting
    tweets_text2 = tweets_text2.replace('', ' ')
    
        #Remove Email Addresses
    tweets_text2 = tweets_text2.replace(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', ' ', regex=True)
    
    tweets_text2 = tweets_text2.str.lower()
        
        
    tweets_text = tweets_text2
    
    return tweets_text


def cross_validation(df, model, features, vectorizer):
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    
    cv = cross_validation.KFold(n=len(df), shuffle=False, n_fold=10)
    
    features = np.asarray(features)
    
    for train_index, test_index in cv:
        
        x_train, x_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = tweets_text[train_index], tweets_text[test_index]
        
        df_features_train = vectorizer.fit.transform(x_train)
        df_features_train = df_features_train.toarray()
        
        #Model trained
        model.fit(df_features_train, y_train)
        
        df_features_test = vectorizer.transform(x_test)
        df_features_test = df_features_test.toarray()
        
        
        #Predict Results
        prediction = model.predict(x_test)
        
        #Get the accuracy 
        
        accuracy = accuracy_score(y_true=y_test, y_pred = prediction)
        recall = recall_score(y_true=y_test, y_pred=prediction)
        precision = precision_score(y_true=y_test, y_pred=prediction)
    
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        
    print("Accuracy Score: ", accuracy_list)
    print("Precision Score ", precision_list)
    print("Recall score ", recall_list)
    

vectorizer = TfidfVectorizer(use_idf=True)
cv = CountVectorizer()

#count_vector = cv.transform(cleaning(tweets_text))

tdidf_vectorizer_vectors = vectorizer.fit_transform(cleaning(tweets_text))
word_count_vector = cv.fit_transform(cleaning(tweets_text))


    
