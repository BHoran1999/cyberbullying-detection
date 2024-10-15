# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import seaborn as sns
import codecs

import random

df = pd.read_csv("cyberbully.csv")

cyberbullying_df = df[df['Insult'] == 1]
noncyberbully_df = df[df['Insult'] == 0]


#Must reset the indexses after seperating the dateset
cyberbullying_df.reset_index(inplace=True, drop=True)
noncyberbully_df.reset_index(inplace=True, drop=True)

#Getting the samples of 20 negative insult
n = len(cyberbullying_df) + 20

#Gathering samples without replacement
random_data = random.sample(range(0, len(noncyberbully_df)), n)
new_cyberbullying_df = noncyberbully_df.iloc[random_data]

#Resetting Index
new_cyberbullying_df.reset_index(inplace=True, drop=True)

#Merging the new dataset
cyberbullying_balance_df = pd.concat([cyberbullying_df, new_cyberbullying_df])

#Shuffling the dataset
cyberbullying_balance_df = cyberbullying_balance_df.sample(frac=1).reset_index(drop=True)

#Save the new dataset

cyberbullying_balance_df.to_csv('cyberbully_balance.csv', index=False)