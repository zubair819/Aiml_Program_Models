import numpy as np
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
Y=np.array(([89],[86],[92]),dtype=float)
X=X/np.amax(X,axis=0)
Y=Y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivatives_sigmoid(x):
    return x*(1-x)
epoch=7000
lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))  
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    hinp1=np.dot(X,wh)
    hinp=hinp1+bh
    hlayer_act=sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp=outinp1+bout
    output=sigmoid(outinp)
    E0=Y-output
    outgrad=derivatives_sigmoid(output)
    d_output=E0*outgrad
    Elt=d_output.dot(wout.T)
    hiddengrad=derivatives_sigmoid(hlayer_act)
    d_hiddenlayer=Elt*hiddengrad
    wout+=hlayer_act.T.dot(d_output) 
print("input:\n"+str(X))    
print("actual output:\n"+str(Y))
print("predicted output:\n",output)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                