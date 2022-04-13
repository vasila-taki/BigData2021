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


porter_stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)

def preprocess(text):
    '''
    Removes numbers and whitespaces and converts to lowercase
    also removes stopwords, words of less than 3 letters and punctuation,
    and perfoms stemming
    '''
    result = re.sub(r'\d+', '', text).lower()
    result = " ".join(result.split())

    text_list = nltk.word_tokenize(result)
    result = ' '.join([porter_stemmer.stem(remove_punctuation(w)) for w in text_list if ((len(w)>2) and (w not in stop_words))])
    
    return result


print("Preprocessing started:")
print(datetime.datetime.now())


df = pd.read_csv("train.csv")
df['Text'] = df['Title'] + df['Content']
df1 = pd.read_csv("test.csv")
df1['Text'] = df1['Title'] + df1['Content']


coun_vect = CountVectorizer(min_df=70,max_df=0.8, binary=True, dtype="bool", lowercase=False)


A1 = df['Text'].values

def f(i, text):
    new_text = preprocess(text)
    return i, new_text

A2 = Parallel(n_jobs=16, max_nbytes='500M')(delayed(f)(i, text) for i,text  in enumerate(A1))
A2 = sorted(A2, key=lambda x: x[0])
A2 = [x[1] for x in A2]

X_train = coun_vect.fit_transform(A2)
X_train = X_train.toarray()

print("shape of X_train:")
print(X_train.shape)


A3 = df1['Text'].values

def f(i, text):
    new_text = preprocess(text)
    return i, new_text

A4 = Parallel(n_jobs=16, max_nbytes='500M')(delayed(f)(i, text) for i,text  in enumerate(A3))
A4 = sorted(A4, key=lambda x: x[0])
A4 = [x[1] for x in A4]

X_test = coun_vect.transform(A4)
X_test = X_test.toarray()

print("shape of X_test:")
print(X_test.shape)


Y_train = df['Target'].values


print("end of data preprocessing:")
print(datetime.datetime.now())


print("Data fitting started:")
print(datetime.datetime.now())


nrst_neigh = KNeighborsClassifier(n_neighbors = 15, metric = 'jaccard', n_jobs=16)
nrst_neigh.fit(X_train,Y_train)

def implement_kNN(test):
   
    indices = nrst_neigh.kneighbors([test], return_distance=False)
    nn_classes = Y_train[indices]
    nn_classes = nn_classes.flatten()
    b = np.asarray(nn_classes,dtype='int')
    return np.bincount(b).argmax()


print("end of fit of training set:")
print(datetime.datetime.now())


x = df1['Id'].values
Ids = np.asarray(x, dtype='int64')

print("Prediction phase started:")
print(datetime.datetime.now())

Y_pred = np.zeros((len(X_test),2),dtype='int')
for i, test in enumerate(X_test):
    if i % 100 == 0: 
        print(i, " ", datetime.datetime.now())
    Y_pred[i] = np.array([Ids[i], implement_kNN(test)])
   
print("end of prediction phase:")
print(datetime.datetime.now())

df3 = pd.DataFrame(data=Y_pred, columns=["Id", "Predicted"])
df3.to_csv(r'1_results_mindf100.csv', index = False)


