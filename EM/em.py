import numpy as np
import pandas as pd
from matplotlib import pyplot  as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

data=pd.read_csv('EM\emcsv.csv')
print("input data and shape")
print(data.shape)
print(data.head())

f1=data['v1'].values
f2=data['v2'].values
x=np.array(list(zip(f1,f2)))

print('x ',x)
print('graph for whole datasets')
plt.scatter(f1, f2,c="black",s=7)
plt.show()

KMeans=KMeans(20,random_state=0)
labels=KMeans.fit(x).predict(x)
print('labels ',labels)
centroids=KMeans.cluster_centers_
print("centroids ",centroids)
plt.scatter(x[:,0],x[:,1],c=labels,s=40,cmap='viridis');
print('graph using kmeans algorithm')
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='#050505');
plt.show()

gmm=GaussianMixture(n_components=3).fit(x)
labels=gmm.predict(x)
probs=gmm.predict_proba(x)
size=10*probs.max(1)**3
print('graph using EM algorithm')
plt.scatter(x[:,0],x[:,1],c=labels,s=size,cmap='viridis');
plt.show()

