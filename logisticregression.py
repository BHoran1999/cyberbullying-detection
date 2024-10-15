# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer

from nltk.metrics.scores import accuracy, precision, recall, f_measure
import collections
from nltk import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import DecisionTreeClassifier
from nltk import SklearnClassifier
from nltk.classify import MaxentClassifier


from sklearn.metrics import accuracy_score

df = pd.read_csv("cyberbully_balance.csv")

tweets_text = df['Comment']
    
        #Remove URLS
#tweets_text = tweets_text.replace(r'\w+:\/\/\S+', '', regex=True)
    
tweets_text = tweets_text.replace(r'https', ' ', regex=True)
#Remove joining of words as soon as new line is made
tweets_text = tweets_text.replace(r'\n', ' ', regex=True)
tweets_text = tweets_text.replace(r'\\n', '', regex=True)
tweets_text = tweets_text.replace(r'\\xc2', ' ', regex=True)
tweets_text = tweets_text.replace(r'\\xa0', ' ', regex=True)
tweets_text = tweets_text.replace(r'_', ' ', regex=True)
tweets_text = tweets_text.replace(r'-', ' ', regex=True)
tweets_text = tweets_text.replace(r'"""', ' ', regex=True)
    
#Remove Twitter Usernames
tweets_text = tweets_text.replace(r'(\A|\s)@(\w+)+[a-zA-Z0-9_\.]', ' ', regex=True)
    
#Remove non-alphabets
tweets_text = tweets_text.replace(r'[^a-zA-Z ]\s?',' ',regex=True)
    
#Whitespace Formatting
tweets_text = tweets_text.replace('', ' ')
    
#Remove Email Addresses
tweets_text = tweets_text.replace(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', ' ', regex=True)
    
tweets_text = tweets_text.str.lower()


Tweet = []
Labels = []

for row in tweets_text:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    clean_words = [word for word in clean_words if word not in english_stops]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)

    for row in df["Insult"]:
        Labels.append(row)


###############################################################################
#Combining the two in order to get bag of words
combined = zip(Tweet, Labels)

#Creating bag of words and dictionary object
def bag_of_words(words):
    return dict([(word, True) for word in words])

#Key, Value Pair into new list for modeling
#Creating a new list for modeling using key and value
Cyberbullying_Data = []
for k, v in combined:
    bag_of_words(k)
    Cyberbullying_Data.append((bag_of_words(k),v))
    
    
#Splitting the data into training and test sets

train_index, test_index = Cyberbullying_Data[0:1000], Cyberbullying_Data[1000:]

logreg_classifier = MaxentClassifier.train(train_index, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)

train_set = collections.defaultdict(set)
test_set = collections.defaultdict(set)

for i, (features, cyberbullying_label) in enumerate(test_index):
    train_set[cyberbullying_label].add(i)
    result = logreg_classifier.classify(features)
    test_set[result].add(i)
    
    
print("Logistic Regression Recall " , recall(train_set[1], test_set[1]))  


from nltk import bigrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

combined = zip(Tweet, Labels)


#Bag of words for bigrams
def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)  
    bigrams = bigram_finder.nbest(score_fn, n)  
    return bag_of_words(bigrams)

Cyberbullying_Data2 = []

for k, v in combined:
    bag_of_bigrams_words(k)
    Cyberbullying_Data2.append((bag_of_bigrams_words(k),v))
    
train_index2, test_index2 = Cyberbullying_Data2[0:1000], Cyberbullying_Data2[1000:]

train_set = collections.defaultdict(set)
test_set = collections.defaultdict(set)
    
logreg_classifier = MaxentClassifier.train(train_index, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)

for i, (features, cyberbullying_label) in enumerate(test_index2):
    train_set[cyberbullying_label].add(i)
    result = logreg_classifier.classify(features)
    test_set[result].add(i)
    
print("Logistic Regression Recall Cyberbullying Recall Bigram: ", recall(test_set[1], train_set[1]))
   
print()
print("Logistic Regression  Recall " , recall(train_set[1], test_set[1]))
print("Logistic Regression  Precision " , precision(train_set[1], test_set[1]))
print("Logistic Regression  F-Measure " , f_measure(train_set[1], test_set[1]))
