# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:17:02 2024

@author: bitm
"""

import csv
a=[]
csvfile=open('CANDIDATE_ELIMINATION/tennis.csv','r')
reader=csv.reader(csvfile)
for row in reader:
    a.append(row)
    print(row)
num_att=len(a[0])-1
print("Initial hypothesis is")
s=['0']*num_att
g=['?']*num_att
print("The most specific: ",s)
print("The most general: ",g)
for j in range((num_att)):
    s[j]=a[0][j]
print("The candidate algorithm\n")
temp=[]
for i in range(len(a)):
    if a[i][num_att]=='yes':
        for j in range(num_att):
            if a[i][j]!=s[j]:
                s[j]='?'
        for j in range(num_att):
            for k in range(1,len(temp)):
                if temp[k][j]!='?' and temp[k][j]!=s[j]:
                    del temp[k]
        print("For instance {0} the hypothesis is s{0}".format(i+1),s)
        if len(temp)==0:
            print("For instance {0} the hypothesis is g{0}".format(i+1),g)
        else:
            print("For instance {0} the hypothesis is g{0}".format(i+1),temp)
    if a[i][num_att]=='no':
        for j in range(num_att):
            if s[j]!=a[i][j] and s[j]!='?':
                g[j]=s[j]
                temp.append(g)
            g=['?']*num_att
        print("For instance {0} the hypothesis is s{0}".format(i+1),s)
        print("For instance {0} the hypothesis is g{0}".format(i+1),temp)
