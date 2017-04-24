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
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('300k.csv', delimiter=',',usecols=["pokemonId", "latitude", "longitude","terrainType","closeToWater","city"])
#restrict dataset to chicago
chicago = df[(df['city'] == 'Chicago')]
y = chicago['pokemonId'].tolist()
x = np.column_stack((chicago['latitude'].tolist(), chicago['longitude'].tolist()))
x1 = np.column_stack( (chicago['latitude'].tolist(), chicago['longitude'].tolist(),chicago['terrainType'].tolist(),chicago['closeToWater'].tolist())    )

#y = df['pokemonId'].tolist()
#x = np.column_stack( (df['latitude'].tolist(),df['longitude'].tolist())  )


#n = 10 #number of splits on kFold (index to split test from train)
#t = 10; #number of desicion trees #n_estimator = number of bags
#mFeatures = 4;#max_features = number of features to draw from x
#mSample = 10;#max_samples = number of samples to draw to place in bag, replace is defaulted to true



print ("\nTree_num/Bags: ", 25)
print("Splits: ", 10)
print ("max feature: ", 4)
print ("max sample: ", 15600)

#NEW YORK
ny = df[(df['city'] == 'New_York')]
ny_x = np.column_stack( (ny['latitude'].tolist(), ny['longitude'].tolist(),ny['terrainType'].tolist(),ny['closeToWater'].tolist())    )
ny_y = ny['pokemonId'].tolist()

#PHEONIX
ph = df[(df['city'] == 'Phoenix')]
ph_x = np.column_stack( (ph['latitude'].tolist(), ph['longitude'].tolist(),ph['terrainType'].tolist(),ph['closeToWater'].tolist())    )
ph_y = ph['pokemonId'].tolist()


t = 25;
seed = 7
mSample = 500
n = 10
mFeatures = 4
kfold = model_selection.KFold(n_splits = n ,random_state = seed)
cart = DecisionTreeClassifier()
num_trees = t

model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
model.fit(x1,y)
trainScore = model.score(x1,y)
testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)


print("Training Score Chicago ", trainScore)
print("Testing Score Chicago ", testingScore.mean())

#NEW YORK
model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
model.fit(ny_x,ny_y)
trainScore_ny = model.score(ny_x,ny_y)
testingScore_ny = model_selection.cross_val_score(model,ny_x,ny_y,cv=kfold)

print("Training Score New York ", trainScore_ny)
print("Testing Score New York ", testingScore_ny.mean())
"""
#Pheonix
model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
model.fit(ph_x,ph_y)
trainScore_ph = model.score(ph_x,ph_y)
testingScore_ph = model_selection.cross_val_score(model,ph_x,ph_y,cv=kfold)

print("Training Score Pheonix ", trainScore_ph)
print("Testing Score Pheonix ", testingScore_ph.mean())
"""


""" This big load (Takes a while procced at your own risk)
nn = 5
t = 50;
seed = 7
mSample = 15600
n = 10
mFeatures = 4
kfold = model_selection.KFold(n_splits = n ,random_state = seed)

cart = KNeighborsClassifier(nn)
cart.fit(x1, y) 
num_trees = t

#Bagging Classifier
model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
model.fit(x1,y)

trainScore = model.score(x1,y)
testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)


print ("\nNN: ", nn)
print("Training Score of NN ", trainScore)
print("Testing Score of NN", testingScore.mean())

t = 50;
seed = 7
mSample = 15600
n = 10
mFeatures = 4
kfold = model_selection.KFold(n_splits = n ,random_state = seed)

cart = DecisionTreeClassifier()
num_trees = t

#Bagging Classifier
model_1 = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
model_1.fit(x1,y)

trainScore_1 = model_1.score(x1,y)
testingScore_1 = model_selection.cross_val_score(model_1,x1,y,cv=kfold)

print ("\nDT: ", nn)
print("Training Score of DT ", trainScore_1)
print("Testing Score of DT", testingScore_1.mean())
"""
    

"""NN 1 - 10
for nn in range (1,10):

    
    t = 1;
    seed = 7
    mSample = 15600
    n = 10
    mFeatures = 4
    kfold = model_selection.KFold(n_splits = n ,random_state = seed)
    
    
    cart = KNeighborsClassifier(nn)
    cart.fit(x1, y) 
    num_trees = t
    
    #Bagging Classifier
    model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
    model.fit(x1,y)
    
    trainScore = model.score(x1,y)
    testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)


    print ("\nNN: ", nn)
    print("Training Score ", trainScore)
    print("Testing Score ", testingScore.mean())
"""

""" Real basic model one used for comparison
t = 1;
seed = 7
mSample = 15600
n = 10
mFeatures = 4
kfold = model_selection.KFold(n_splits = n ,random_state = seed)
cart = DecisionTreeClassifier()

num_trees = t

#Bagging Classifier
model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
model.fit(x1,y)

trainScore = model.score(x1,y)
testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)


print ("\nTree_num/Bags: ", t)
print("Splits: ", n)
print ("max feature: ", mFeatures)
print ("max sample: ", mSample)
print("Training Score ", trainScore)
print("Testing Score ", testingScore.mean())
"""


""" How features effect the data
t = 5;
seed = 7
mSample = 15600
n = 10
for mFeatures in range (2,5):
    kfold = model_selection.KFold(n_splits = n ,random_state = seed)
    cart = DecisionTreeClassifier()
    
    num_trees = t
    
    #Bagging Classifier
    model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
    model.fit(x1,y)
    
    trainScore = model.score(x1,y)
    testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)
    
    
    print ("\nTree_num/Bags: ", t)
    print("Splits: ", n)
    print ("max feature: ", mFeatures)
    print ("max sample: ", mSample)
    print("Training Score ", trainScore)
    print("Testing Score ", testingScore.mean())
"""


""" How splits affect the data
t = 10;
mFeatures = 2;
seed = 7
mSample = 15600

for n in range (10,1000,10):
    kfold = model_selection.KFold(n_splits = n ,random_state = seed)
    cart = DecisionTreeClassifier()
    
    num_trees = t
    
    #Bagging Classifier
    model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
    model.fit(x1,y)
    
    trainScore = model.score(x1,y)
    testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)
    
    
    print ("\nTree_num/Bags: ", t)
    print("Splits: ", n)
    print ("max feature: ", mFeatures)
    print ("max sample: ", mSample)
    print("Training Score ", trainScore)
    print("Testing Score ", testingScore.mean())
"""

""" Increase number of bag with sample at 15600
n = 2;
mFeatures = 2;
seed = 7
mSample = 15600

kfold = model_selection.KFold(n_splits = n ,random_state = seed)
cart = DecisionTreeClassifier()

for t in range (10,1000,10):

    num_trees = t
    
    #Bagging Classifier
    model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
    model.fit(x1,y)
    
    trainScore = model.score(x1,y)
    testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)
    
    
    print ("\nTree_num/Bags: ", t)
    print("Splits: ", n)
    print ("max feature: ", mFeatures)
    print ("max sample: ", mSample)
    print("Training Score ", trainScore)
    print("Testing Score ", testingScore.mean())
"""

""" Increase number of samples with bag at 1
n = 2;
mFeatures = 2;
seed = 7
t = 1

kfold = model_selection.KFold(n_splits = n ,random_state = seed)
cart = DecisionTreeClassifier()

for mSample in range (10,10000,100):

    num_trees = t
    
    #Bagging Classifier
    model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
    model.fit(x1,y)
    
    trainScore = model.score(x1,y)
    testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)
    
    
    print ("\nTree_num/Bags: ", t)
    print("Splits: ", n)
    print ("max feature: ", mFeatures)
    print ("max sample: ", mSample)
    print("Training Score ", trainScore)
    print("Testing Score ", testingScore.mean())
"""

""" Loop Testing
for t in range (1,20,5):
    for n in range (2,100,2):
        for mFeatures in range (2,4):
            for mSample in range(10,100,10):             
                seed = 7
                
                #How to split the data for 
                kfold = model_selection.KFold(n_splits = n ,random_state = seed)
                cart = DecisionTreeClassifier()
                num_trees = t
                
                #Bagging Classifier
                model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees, max_samples = mSample, max_features = mFeatures,random_state = seed,bootstrap = True)
                model.fit(x1,y)
                
                m1 = RandomForestClassifier(n_estimators=num_trees, max_features=3)
                trainScore = model.score(x1,y)
                testingScore = model_selection.cross_val_score(model,x1,y,cv=kfold)
                               
                print ("\nTree_num/Bags: ", t)
                print("Splits: ", n)
                print ("max feature: ", mFeatures)
                print ("max sample: ", mSample)
                print("Training Score ", trainScore)
                print("Testing Score ", testingScore.mean())
"""


""" Intial Test
#Notes: Increaseing the n_splits gives a better testing score
# increasing the num_trees improves training score
#Change or remove the loop values 
for t in range(10,100,10):
    for n in range(2,10):   #Range must start at two

        #K fold testing
        kfold = model_selection.KFold(n_splits = n ,random_state = seed)
        
        cart = DecisionTreeClassifier()
        
        num_trees = t
        
        #n_estimator = number of bags
        #max_samples = number of samples to draw to place in bag, replace default = true
        #max_features = number of features to draw from x
        
        
        model = BaggingClassifier(base_estimator = cart,n_estimators = num_trees,random_state = seed,bootstrap = True)
        model.fit(x,y)
        trainScore = model.score(x,y)
        
        testingScore = model_selection.cross_val_score(model,x,y,cv=kfold)
        
        print("\nSplits: ", n)
        print ("Tree_num: ", t)
        print("Training Score ", trainScore)
        print("Testing Score ", testingScore.mean())

"""