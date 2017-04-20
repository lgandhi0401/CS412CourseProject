# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:14:56 2017

@author: Joe & Lakulish
Resource: http://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../input/300k.csv', delimiter=',',usecols=["pokemonId", "latitude", "longitude","city"])
#restrict dataset to chicago
chicago = df[(df['city'] == 'Chicago')]
y = chicago['pokemonId'].tolist()
x = np.column_stack((chicago['latitude'].tolist(), chicago['longitude'].tolist()))

#y = df['pokemonId'].tolist()
#x = np.column_stack( (df['latitude'].tolist(),df['longitude'].tolist())  )

seed = 7

#Notes: Increaseing the n_splits gives a better testing score
# increasing the num_trees improves training score
#Change or remove the loop values 
for t in range(10,100,10):
    for n in range(2,10):   #Range must start at two

        #K fold testing
        kfold = model_selection.KFold(n_splits = n ,random_state = seed)
        
        cart = DecisionTreeClassifier()
        
        num_trees = t
        
        model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees,random_state = seed,bootstrap = True)
        model.fit(x,y)
        trainScore = model.score(x,y)
        
        testingScore = model_selection.cross_val_score(model,x,y,cv=kfold)
        
        print("\nSplits: ", n)
        print ("Tree_num: ", t)
        print("Training Score ", trainScore)
        print("Testing Score ", testingScore.mean())

