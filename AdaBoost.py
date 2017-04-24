# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:51:57 2017

@author: Joe & Lakulish
Resources: http://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

#read csv using id, latitude and longitude and city columns
df = pd.read_csv('../input/300k.csv', delimiter=',',usecols=["pokemonId", "latitude", "longitude","city"])

#restrict dataset to chicago
chicago = df[(df['city'] == 'Chicago')]
Y = chicago['pokemonId'].tolist()
X = np.column_stack((chicago['latitude'].tolist(), chicago['longitude'].tolist()))

seed = 7

num_trees = 30

kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X,Y)

trainScore = model.score(X,Y)
testScore = model_selection.cross_val_score(model, X, Y, cv=kfold)


print('Training score',trainScore)
print('Test score',testScore.mean())