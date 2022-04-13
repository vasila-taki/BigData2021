import numpy as np
import pandas as pd
import re
import collections
import math
import pyspark.sql.functions as F
import pyspark.sql.types as T
import json
import datetime

def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

df = pd.read_csv("dtw_test.csv")

np_M = df.values
strA = np_M[:, [1]]
strB = np_M[:, [2]]

A = []
B = []

for i in range(len(np_M)):
  A.append(parse_embedding_from_string(strA[i][0]))
  B.append(parse_embedding_from_string(strB[i][0]))


def calculate_euclidean(a, b):

  dist = a - b

  return np.linalg.norm(dist)

def DTW_Distance(a, b):
  
  DTW = np.ones((len(a)+1,len(b)+1), dtype='float') * math.inf
  
  DTW[0][0] = 0

  for i in range(len(a)):
    for j in range(len(b)):
      cost = calculate_euclidean(a[i],b[j])
      minimum = min(DTW[i][j+1], DTW[i+1][j], DTW[i][j])
      DTW[i+1][j+1] = cost + minimum    
            
  return DTW[len(a)][len(b)]

x = df['id'].values
Ids = np.asarray(x, dtype='int64')


print("Calculation of distancies started:")
print(datetime.datetime.now())


Y_calc = np.zeros((len(np_M),2),dtype='float64')
for i in range(len(np_M)):
    
    Y_calc[i] = np.array([Ids[i], DTW_Distance(A[i], B[i])])
    print(i, " ", datetime.datetime.now())

print("end of calculation of distancies:")
print(datetime.datetime.now())


df3 = pd.DataFrame(data=Y_calc, columns=["id", "distance"])

convert_dict = {"id" : int, "distance" : float}
df3 = df3.astype(convert_dict)

df3.to_csv(r'dtw.csv', index = False)

