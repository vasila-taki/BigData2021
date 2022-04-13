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


print("Build phase started:")
print(datetime.datetime.now())


perm = 64
hashed = []

for i in range(len(X_train)):
  m = MinHash(num_perm=perm)

  for d in set(coun_vect.inverse_transform(X_train[i].reshape(1, -1))[0].tolist()):
    m.update(d.encode('utf8'))

  hashed.append(m)

# Create LSH index
lsh = MinHashLSH(threshold=0.8, num_perm=perm)

for i in range(len(hashed)):
  lsh.insert(str(i), hashed[i])

hashed_test = []

for i in range(len(X_test)):
  m = MinHash(num_perm=perm)

  for d in set(coun_vect.inverse_transform(X_test[i].reshape(1, -1))[0].tolist()):
    m.update(d.encode('utf8'))

  hashed_test.append(m)

print("end of build phase:")
print(datetime.datetime.now())


def implement_kNN(hashed_test, test, X_tr, Y_tr, k):

    nrst_neigh = KNeighborsClassifier(n_neighbors = k, metric = 'jaccard')
    
    keys_over_threshold = lsh.query(hashed_test)
    keys_over_threshold = np.asarray(keys_over_threshold, dtype='int')

    if(len(keys_over_threshold) >= k):
        candidates = []
        candidates_labels = np.zeros(len(keys_over_threshold),dtype = int)

        for i in range(len(keys_over_threshold)):
            candidates.append(X_tr[keys_over_threshold[i]])
            candidates_labels[i] = Y_tr[keys_over_threshold[i]]
    
        nrst_neigh.fit(candidates,candidates_labels)
    
        indices = nrst_neigh.kneighbors([test], return_distance=False)
    
        nn_classes = Y_train[indices]
        nn_classes = nn_classes.flatten()
        b = np.asarray(nn_classes,dtype='int')
        
        c=1
        return np.bincount(b).argmax(), c
    else:
        nrst_neigh.fit(X_train,Y_train)

        indices = nrst_neigh.kneighbors([test], return_distance=False)
        nn_classes = Y_train[indices]
        nn_classes = nn_classes.flatten()
        b = np.asarray(nn_classes,dtype='int')

        c=0        
        return np.bincount(b).argmax(), c


x = df0['id'].values
Ids = np.asarray(x, dtype='int64')


print("Query phase started:")
print(datetime.datetime.now())

impl = 0
Y_pred = np.zeros((len(X_test),2),dtype='int')
for i, test in enumerate(X_test):
    if i % 100 == 0: 
        print(i, " ", datetime.datetime.now())
    pred, k = implement_kNN(hashed_test[i], test, X_train, Y_train, 15)
    impl = impl + k
    Y_pred[i] = np.array([Ids[i], pred])

print("end of Query phase:")
print(datetime.datetime.now())

print("LSH implementations:")
print(impl)

df3 = pd.DataFrame(data=Y_pred, columns=["id", "sentiment"])
df3.to_csv(r'ac_df200_perm64_thres0.8.csv', index = False)


