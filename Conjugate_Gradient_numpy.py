import numpy as np

CG_TOL=1e-6
CG_MAX_ITER=50
NEWTON_MAX_ITERS=100
EPSILON=1e-9

#------------------------------------------------
#DUMMY FUNCTIONS (UNDEFINED ERRORS WERE ANNOYING)

def sigmoid(z):
    return

def gradient(X,y,w):
    return

def loss(X,y,w):
    return

#------------------------------------------------

def hessian_vector_product(X,w,p):
    z=X@w
    s=sigmoid(z)
    D=s*(1-s)
    return X.T@(D*(X@p))

def conjugate_gradient(X,w,g,max_iters=CG_MAX_ITER):
    dw=np.zeros_like(w)
    r=g.copy()
    p=r.copy()
    Hp=hessian_vector_product(X,w,p)
    alpha=np.dot(r,r)/np.dot(p,Hp)
    dw=dw+alpha*p
    for k in range(1,max_iters):
        r_old=r.copy()
        r=r_old-alpha*Hp
        w_CG=np.dot(r,r)/np.dot(r_old,r_old)
        p=r+w_CG*p
        Hp=hessian_vector_product(X,w,p)
        alpha=np.dot(r,r)/np.dot(p,Hp)
        dw=dw+alpha*p
        if np.linalg.norm(dw)/np.linalg.norm(w+EPSILON) < CG_TOL:
            break
    return dw

def newton_cg(X,y,w,max_iters=NEWTON_MAX_ITERS):
    losses=[]
    for iter in range(0,max_iters):
        g=gradient(X,y,w)
        dw=conjugate_gradient(X,w,g)
        w=w-dw
        losses.append(loss(X,y,w))
    return losses,w