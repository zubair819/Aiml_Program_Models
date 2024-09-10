import csv
num_attributes=6
a=[]
print("\n The given training data set \n")
with open ('FIND_S/tennis.csv') as csvfile:
    reader=csv.reader(csvfile)
    reader=list(reader)
for row in reader:
    a.append(row)
    print(row)
print("The initial values of hypothesis ")
hypothesis=['0']*num_attributes
print(hypothesis)
for j in range(0,num_attributes):
    hypothesis[j]=a[0][j]
for i in range(0,len(a)):
    if(a[i][num_attributes]=='yes'):
        for j in range(0,num_attributes):
            if(a[i][j]!=hypothesis[j]):
                hypothesis[j]='?'
            else:
                hypothesis[j]=a[i][j]
    print("For training instance no:",i,"the hypothesis is",hypothesis)
print("The maximally specific hypothesis is ",hypothesis)