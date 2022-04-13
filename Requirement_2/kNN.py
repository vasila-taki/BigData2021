import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from os import path
import string
import re
import collections
import nltk
import matplotlib.pyplot as plt
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from nltk import word_tokenize 
import datetime
import json


porter_stemmer = PorterStemmer()

def normalize(X, average):
    for i in range(len(X)):
        if(X[i] >= average[i]):
            X[i] = 1
        else:
            X[i] = 0
    return X


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)

def preprocess(text):
    '''
    converts to lowercase, removes punctuation,
    and performs stemming
    '''
    result = text.lower()

    text_list = nltk.word_tokenize(result)
    result = ' '.join([porter_stemmer.stem(remove_punctuation(w)) for w in text_list])
    
    return result


print("Preprocessing started:")
print(datetime.datetime.now())

df = pd.read_csv("imdb_train.csv")
df0 = pd.read_csv("imdb_test.csv")

coun_vect = CountVectorizer(min_df=200, dtype='int8', lowercase=False)

A1 = df['review'].values

def f1(i, text):
    new_text = preprocess(text)
    return i, new_text

A2 = Parallel(n_jobs=16, max_nbytes='500M')(delayed(f1)(i, text) for i,text  in enumerate(A1))
A2 = sorted(A2, key=lambda x: x[0])
A2 = [x[1] for x in A2]

Xtrain = coun_vect.fit_transform(A2)
Xtrain = Xtrain.toarray()

print("shape of Xtrain:")
print(Xtrain.shape)


av = np.sum(Xtrain, axis=0) / len(Xtrain)


A3 = df0['review'].values

A4 = Parallel(n_jobs=16, max_nbytes='500M')(delayed(f1)(i, text) for i,text in enumerate(A3))
A4 = sorted(A4, key=lambda x: x[0])
A4 = [x[1] for x in A4]

Xtest = coun_vect.transform(A4)
Xtest = Xtest.toarray()

print("shape of Xtest:")
print(Xtest.shape)


def f2(i, row):
    new_row = normalize(row,av)
    return i, new_row

X_train = Parallel(n_jobs=16, max_nbytes='500M')(delayed(f2)(i, row) for i,row in enumerate(Xtrain))
X_train = sorted(X_train, key=lambda x: x[0])
X_train = [x[1] for x in X_train]
    
X_train = np.array(X_train, dtype=bool)


X_test = Parallel(n_jobs=16, max_nbytes='500M')(delayed(f2)(i, row) for i,row in enumerate(Xtest))
X_test = sorted(X_test, key=lambda x: x[0])
X_test = [x[1] for x in X_test]

X_test = np.array(X_test, dtype=bool)


Y_train = df['sentiment'].values

print("end of data preprocessing:")
print(datetime.datetime.now())


print("Data fitting started:")
print(datetime.datetime.now())


nrst_neigh = KNeighborsClassifier(n_neighbors = 15, metric = 'jaccard')
nrst_neigh.fit(X_train,Y_train)

def implement_kNN(test):
    indices = nrst_neigh.kneighbors([test], return_distance=False)
    nn_classes = Y_train[indices]
    nn_classes = nn_classes.flatten()
    b = np.asarray(nn_classes,dtype='int')
    return np.bincount(b).argmax()
  

print("end of fit of training set:")
print(datetime.datetime.now())


x = df0['id'].values
Ids = np.asarray(x, dtype='int64')


print("Query phase started:")
print(datetime.datetime.now())


Y_pred = np.zeros((len(X_test),2),dtype='int')
for i, test in enumerate(X_test):
    if i % 100 == 0: 
        print(i, " ", datetime.datetime.now())
    Y_pred[i] = np.array([Ids[i], implement_kNN(test)])


print("end of Query phase:")
print(datetime.datetime.now())


df3 = pd.DataFrame(data=Y_pred, columns=["id", "sentiment"])
df3.to_csv(r'ac_df200_KNN.csv', index = False)
