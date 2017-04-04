# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#read csv using id, latitude and longitude and city columns
df = pd.read_csv('../input/300k.csv', delimiter=',',usecols=["pokemonId", "latitude", "longitude","city"])

#restrict dataset to chicago
chicago = df[(df['city'] == 'Chicago')]
y = chicago['pokemonId'].tolist()
X = np.column_stack((chicago['latitude'].tolist(), chicago['longitude'].tolist()))

#split data into test/train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#run NN with k=1
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(1)
neigh.fit(X_train, y_train) 

#predict placement for X[3]
print(neigh.predict([X[3]]))
#get proability array for X[3]
print(neigh.predict_proba([X[3]]))
print(y[3])
#get accuracy for training/testing sets
print(neigh.score(X_train,y_train))
print(neigh.score(X_test,y_test))
