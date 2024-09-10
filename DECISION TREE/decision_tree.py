# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:51:43 2024

@author: bitm
"""

import pandas as pd
from collections import Counter 
import math
tennis = pd.read_csv('DECISION TREE\playtennis.csv')
print("\n given play tennis data set:\n\n",tennis)
def entropy(alist):
    c=Counter(x for x in alist)
    instances = len(alist)
    prob = [x / instances for x in c.values()]
    return sum([-p*math.log(p,2)for p in prob])

def information_gain(d, split, target):
    splitting =d.groupby(split)
    n = len(d.index)
    agent = splitting.agg({target:[entropy, lambda x: len(x)/n]})[target]
    agent.columns=['entropy','observations']
    newentropy=sum(agent['entropy']*agent['observations'])
    oldentropy=entropy(d[target])
    return oldentropy-newentropy
 
def id3(sub,target,names):
    count =Counter(x for x in sub[target])
    print(count)
    if len(count)==1:
        return next(iter(count))
    else:
        gain = [information_gain(sub,attr,target) for attr in names]
        print("gain=",gain)
        maximum = gain.index(max(gain))
        best = names[maximum]
        print("best attribute:",best)
        tree = {best:{}}
        remaining = [i for i in names if i!=best]
        for val,subset in sub.groupby(best):
            subtree=id3(subset,target,remaining)
            tree[best][val] =subtree
        return tree
names = list(tennis.columns)
print("list of attributes",names)
names.remove('playtennis')
print("predicting attributes:",names)
tree=id3(tennis,'playtennis',names)
print("\n\n the resultant decision tree is :\n")
print(tree)
 