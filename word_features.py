# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:55:04 2022

@author: bhbri
"""

from cyberbullying.py import cleaning
from evaluation.py import evaluationModel, evaluate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import seaborn as sns
import codecs

from sklearn import neighbors

from sklearn.metrics import  accuracy_score, recall_score, precision_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.models.keyedvectors as Word2Vec

import nltk

import sys

df = pd.read_csv("cyberbully_balance.csv")

x = cleaning(df['Comment'])
y = df['Insult']

#This function uses each word as a feature

def wordFeatureVector(df, model, num_of_features):
    
    #Initializing the empty numpy array in word features
    featureVector = np.zeros((num_of_features,), dtype="float32")
    
    for i, word in enumerate(df.split()):
        
        if i == len(featureVector):
            break
        
        if word in model:
            feature_word = np.mean(model[word])
            
        else:
            feature_word = -1.0 
            
        featureVector[i] = feature_word
        
    return featureVector


def commentFeatureVector(df, model, num_features):
    
    #Initizing a counter
    i = 0
    
    #Initializing a empty numpy array
    commentFeatureVectorReview = np.zeros((len(df), num_features), dtype="float32")
    
    for tweets in df:
        
        commentFeatureVectorReview[i] = wordFeatureVector(df, model, num_features)
        
        i += 1
        
    return commentFeatureVectorReview

max_amount_of_words = 200

data = commentFeatureVector(x, model, max_amount_of_words)

evaluate(x, y)

