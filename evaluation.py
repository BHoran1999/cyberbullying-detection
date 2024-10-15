# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:16:00 2022

@author: bhbri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import seaborn as sns
import codecs

from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn.metrics import accuracy_score, recall_score, precision_score, log_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from sklearn import RandomForestClassifier
from sklearn import LogisticRegression
from sklearn import SVC

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

import sys

import time

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



def evaluationModel(model_name, model, x, y, skv):
    
    print("Classification Name: " + model_name + " Starts\n")
    
    accuracy_list = []
    precision_list = []
    fpr_array = []
    log_array = []
    time_array = []
    
    for train_index, test_index in skv.split(x, y):
        
        
        #Splitting the training and test dataset
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        #Training the Model
        model.fit(x_train , y_train)
        
        
        #Calculating the execution time
        
        start_time = time.time()
        
        result = model.predict(x_test)
        
        finish_time = time.time()
        
        execution_time = finish_time - start_time
        
        if model_name == "SVM":
            
            log = LogisticRegression()
            log.fit(x_train, y_train)
            
            y_scores = log.predict.proba(x_test)
            
        else:
            
            y_scores = model.predict.proba(x_test)
            
        
        accuracy = accuracy_score(y_test, result)
        #precision = tp/(tp+fp)
        log_score = log_loss(y_test, y_scores[:, 1])
        #fpr = fp/(fp+tn)
        
        accuracy_list.append(accuracy)
       # precision_list.append(precision)
        log_array.append(log_score)
        time_array.append(execution_time)
        
    #Mean Results
    mean_accuracy = np.mean(accuracy_list)
    mean_precision = np.mean(precision_list)
    mean_log = np.mean(log_array)
    mean_time = np.mean(time_array)
    
    #Display Results
    
    print("Mean Accuracy: %0.3f (+/- %0.3f) \n" % (mean_accuracy))
    print("Mean Precision: %0.3f (+/- %0.3f) \n" % (mean_precision))
    print("Mean Log Loss Score: %0.3f (+/- %0.3f \n" % (mean_log))
    print("Mean Execution Score: %0.3f (+/- %0.3f \n" % (mean_log))
    

def evaluation(x, y):
    
    #List of models
    rf = RandomForestClassifier()
    svm = SVC()
    
    #Storing the models into the dictionary for loop to go through
    
    models = {"RF": rf, "SVM": svm}
    
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    
    for key, model in models.items():
        
        evaluationModel(key, model, x, y, skf)
    
    