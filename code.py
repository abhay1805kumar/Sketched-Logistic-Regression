import time
import numpy as np

EPSILON=1e-9

def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1.0/(1.0+np.exp(-z))

def loss(X,y,w,lam=1e-4):
    z=y*np.sum(X*w,axis=1)
    data=np.mean(np.logaddexp(0.0,-z))
    reg=0.5*lam*np.dot(w,w)
    return data+reg

def gradient(X,y,w,lam=1e-4):
    z=y*np.sum(X*w,axis=1)
    s=sigmoid(z)
    u=y*(1.0-s)
    g=-np.sum(X*u[:,None],axis=0)/len(y)
    return g+lam*w

def hessian_vector_product(X,y,w,v,lam=1e-4):
    z=y*np.sum(X*w,axis=1)
    s=sigmoid(z)
    D=s*(1.0-s)
    Xv=np.sum(X*v,axis=1)
    Hv=np.sum(X*(D*Xv)[:,None],axis=0)/len(y)
    return Hv+lam*v

def hessian_diagonal(X,y,w,lam=1e-4):
    z=y*np.sum(X*w,axis=1)
    s=sigmoid(z)
    D=s*(1.0-s)
    d=np.sum((X**2)*D[:,None],axis=0)/len(y)
    return d+lam

def gradient_descent(X,y,w,lr=0.1,iters=100,lam=1e-4):
    losses=[]
    times=[]
    start=time.perf_counter()
    for i in range(iters):
        g=gradient(X,y,w,lam)
        w=w-lr*g
        losses.append(loss(X,y,w,lam))
        times.append(time.perf_counter()-start)
    return losses,w,times

def load_spambase(path="spambase/spambase.data"):
    data=np.loadtxt(path,delimiter=",")
    X=data[:,:-1]
    y=data[:,-1]
    y=2*y-1
    mu=np.mean(X,axis=0)
    sd=np.std(X,axis=0)
    sd[sd<EPSILON]=1.0
    X=(X-mu)/sd
    return X,y

if __name__=="__main__":
    X,y=load_spambase()
    w=np.zeros(X.shape[1])
    losses,w,_=gradient_descent(X,y,w,lr=0.1,iters=20)
    print("gd final loss:",losses[-1])
