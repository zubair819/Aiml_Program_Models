         # -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:14:14 2024

@author: bitm
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris=load_iris()
x=iris.data
y=iris.target
print(x[:5],y[:5])

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4,random_state=1)

print(iris.data.shape)
print(len(xtrain))
print(len(ytest))

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, ytrain)

pred=knn.predict(xtest)
print("Accuracy", metrics.accuracy_score(ytest, pred))

ytestn = [iris.target_names[i] for i in ytest]
predn = [iris.target_names[i] for i in pred]
print("PREDICTED        ACTUAL")
for i in range((len(pred))):
    print(i," ",predn[i],' ',ytestn[i])