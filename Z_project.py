import time
import pathlib
import importlib.util
import numpy as np

base_path=pathlib.Path(__file__).resolve().parent
code_path=base_path/"code.py"
spec=importlib.util.spec_from_file_location("project_code",code_path)
project_code=importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_code)

loss=project_code.loss
gradient=project_code.gradient
hessian_vector_product=project_code.hessian_vector_product
hessian_diagonal=project_code.hessian_diagonal
gradient_descent=project_code.gradient_descent
load_spambase=project_code.load_spambase

EPSILON=1e-9
CG_TOL=1e-6
CG_MAX_ITER=50
NEWTON_ITERS=20

def hessian(X,y,w,lam=1e-4):
    z=y*np.sum(X*w,axis=1)
    s=1.0/(1.0+np.exp(-np.clip(z,-500,500)))
    D=s*(1.0-s)
    H=np.einsum("ni,nj->ij",X,D[:,None]*X)/len(y)
    H=H+lam*np.eye(X.shape[1])
    return H

def line_search(X,y,w,p,g,lam=1e-4,a=1.0,b=0.5,c=1e-4):
    f0=loss(X,y,w,lam)
    gg=np.dot(g,p)
    alpha=a
    while alpha>1e-8:
        w_new=w-alpha*p
        f_new=loss(X,y,w_new,lam)
        if f_new<=f0-c*alpha*gg:
            return alpha
        alpha=b*alpha
    return alpha

def newton_method(X,y,w,iters=NEWTON_ITERS,lam=1e-4,use_line_search=True):
    losses=[]
    times=[]
    start=time.perf_counter()
    for k in range(iters):
        g=gradient(X,y,w,lam)
        H=hessian(X,y,w,lam)
        p=np.linalg.solve(H,g)
        alpha=1.0
        if use_line_search:
            alpha=line_search(X,y,w,p,g,lam)
        w=w-alpha*p
        losses.append(loss(X,y,w,lam))
        times.append(time.perf_counter()-start)
    return losses,w,times

def conjugate_gradient(X,y,w,g,lam=1e-4,tol=CG_TOL,max_iters=CG_MAX_ITER):
    step=np.zeros_like(w)
    r=g.copy()
    p=r.copy()
    rr=np.dot(r,r)
    if np.sqrt(rr)<tol:
        return step,0
    for k in range(max_iters):
        Hp=hessian_vector_product(X,y,w,p,lam)
        den=np.dot(p,Hp)
        if abs(den)<EPSILON:
            break
        alpha=rr/den
        step=step+alpha*p
        r=r-alpha*Hp
        rr_new=np.dot(r,r)
        if np.sqrt(rr_new)<=tol*(np.linalg.norm(g)+EPSILON):
            return step,k+1
        beta=rr_new/(rr+EPSILON)
        p=r+beta*p
        rr=rr_new
    return step,k+1

def newton_cg(X,y,w,iters=NEWTON_ITERS,lam=1e-4,tol=CG_TOL,max_cg=CG_MAX_ITER,use_line_search=True):
    losses=[]
    times=[]
    cg_steps=[]
    start=time.perf_counter()
    for k in range(iters):
        g=gradient(X,y,w,lam)
        p,cg_iter=conjugate_gradient(X,y,w,g,lam,tol,max_cg)
        alpha=1.0
        if use_line_search:
            alpha=line_search(X,y,w,p,g,lam)
        w=w-alpha*p
        losses.append(loss(X,y,w,lam))
        times.append(time.perf_counter()-start)
        cg_steps.append(cg_iter)
    return losses,w,times,cg_steps

def diagonal_preconditioner(X,y,w,lam=1e-4):
    d=hessian_diagonal(X,y,w,lam)
    return 1.0/(d+EPSILON)

def preconditioned_conjugate_gradient(X,y,w,g,lam=1e-4,tol=CG_TOL,max_iters=CG_MAX_ITER):
    step=np.zeros_like(w)
    r=g.copy()
    M=diagonal_preconditioner(X,y,w,lam)
    z=M*r
    p=z.copy()
    rz=np.dot(r,z)
    if np.sqrt(np.dot(r,r))<tol:
        return step,0
    for k in range(max_iters):
        Hp=hessian_vector_product(X,y,w,p,lam)
        den=np.dot(p,Hp)
        if abs(den)<EPSILON:
            break
        alpha=rz/den
        step=step+alpha*p
        r=r-alpha*Hp
        if np.sqrt(np.dot(r,r))<=tol*(np.linalg.norm(g)+EPSILON):
            return step,k+1
        z=M*r
        rz_new=np.dot(r,z)
        beta=rz_new/(rz+EPSILON)
        p=z+beta*p
        rz=rz_new
    return step,k+1

def newton_pcg(X,y,w,iters=NEWTON_ITERS,lam=1e-4,tol=CG_TOL,max_cg=CG_MAX_ITER,use_line_search=True):
    losses=[]
    times=[]
    cg_steps=[]
    start=time.perf_counter()
    for k in range(iters):
        g=gradient(X,y,w,lam)
        p,cg_iter=preconditioned_conjugate_gradient(X,y,w,g,lam,tol,max_cg)
        alpha=1.0
        if use_line_search:
            alpha=line_search(X,y,w,p,g,lam)
        w=w-alpha*p
        losses.append(loss(X,y,w,lam))
        times.append(time.perf_counter()-start)
        cg_steps.append(cg_iter)
    return losses,w,times,cg_steps

def gaussian_sketch(n,m):
    return np.random.randn(m,n)/np.sqrt(m)

def sketched_hessian_vector_product(X,y,w,v,S,lam=1e-4):
    z=y*np.sum(X*w,axis=1)
    s=1.0/(1.0+np.exp(-np.clip(z,-500,500)))
    D=s*(1.0-s)
    rt=np.sqrt(D)
    Xv=np.sum(X*v,axis=1)
    u=rt*Xv
    u=np.einsum("mn,n->m",S,u)
    u=np.einsum("mn,m->n",S,u)
    Hv=np.sum(X*(rt*u)[:,None],axis=0)/len(y)
    return Hv+lam*v

def sketched_conjugate_gradient(X,y,w,g,S,lam=1e-4,tol=CG_TOL,max_iters=CG_MAX_ITER):
    step=np.zeros_like(w)
    r=g.copy()
    p=r.copy()
    rr=np.dot(r,r)
    if np.sqrt(rr)<tol:
        return step,0
    for k in range(max_iters):
        Hp=sketched_hessian_vector_product(X,y,w,p,S,lam)
        den=np.dot(p,Hp)
        if abs(den)<EPSILON:
            break
        alpha=rr/den
        step=step+alpha*p
        r=r-alpha*Hp
        rr_new=np.dot(r,r)
        if np.sqrt(rr_new)<=tol*(np.linalg.norm(g)+EPSILON):
            return step,k+1
        beta=rr_new/(rr+EPSILON)
        p=r+beta*p
        rr=rr_new
    return step,k+1

def sketched_newton(X,y,w,m,iters=NEWTON_ITERS,lam=1e-4,tol=CG_TOL,max_cg=CG_MAX_ITER,use_line_search=True):
    losses=[]
    times=[]
    cg_steps=[]
    start=time.perf_counter()
    for k in range(iters):
        g=gradient(X,y,w,lam)
        S=gaussian_sketch(len(y),m)
        p,cg_iter=sketched_conjugate_gradient(X,y,w,g,S,lam,tol,max_cg)
        alpha=1.0
        if use_line_search:
            alpha=line_search(X,y,w,p,g,lam)
        w=w-alpha*p
        losses.append(loss(X,y,w,lam))
        times.append(time.perf_counter()-start)
        cg_steps.append(cg_iter)
    return losses,w,times,cg_steps

def run_all():
    X,y=load_spambase()
    d=X.shape[1]
    w0=np.zeros(d)

    gd_losses,gd_w,gd_times=gradient_descent(X,y,w0.copy(),lr=0.1,iters=100,lam=1e-4)

    newton_losses,newton_w,newton_times=newton_method(X,y,w0.copy(),iters=12,lam=1e-4,use_line_search=True)
    cg_losses,cg_w,cg_times,cg_steps=newton_cg(X,y,w0.copy(),iters=12,lam=1e-4,tol=1e-6,max_cg=50,use_line_search=True)
    pcg_losses,pcg_w,pcg_times,pcg_steps=newton_pcg(X,y,w0.copy(),iters=12,lam=1e-4,tol=1e-6,max_cg=50,use_line_search=True)
    m1=d
    m2=2*d
    sketch_losses_1,sketch_w_1,sketch_times_1,sketch_steps_1=sketched_newton(X,y,w0.copy(),m1,iters=12,lam=1e-4,tol=1e-6,max_cg=50,use_line_search=True)
    sketch_losses_2,sketch_w_2,sketch_times_2,sketch_steps_2=sketched_newton(X,y,w0.copy(),m2,iters=12,lam=1e-4,tol=1e-6,max_cg=50,use_line_search=True)

    out={
        "gd":{"losses":gd_losses,"times":gd_times,"w":gd_w},
        "newton":{"losses":newton_losses,"times":newton_times,"w":newton_w},
        "newton_cg":{"losses":cg_losses,"times":cg_times,"cg_steps":cg_steps,"w":cg_w},
        "newton_pcg":{"losses":pcg_losses,"times":pcg_times,"cg_steps":pcg_steps,"w":pcg_w},
        "sketched_newton_d":{"losses":sketch_losses_1,"times":sketch_times_1,"cg_steps":sketch_steps_1,"w":sketch_w_1,"m":m1},
        "sketched_newton_2d":{"losses":sketch_losses_2,"times":sketch_times_2,"cg_steps":sketch_steps_2,"w":sketch_w_2,"m":m2},
    }

    print("gd final loss:",gd_losses[-1])
    print("newton final loss:",newton_losses[-1])
    print("newton cg final loss:",cg_losses[-1])
    print("newton pcg final loss:",pcg_losses[-1])
    print("sketched newton d final loss:",sketch_losses_1[-1])
    print("sketched newton 2d final loss:",sketch_losses_2[-1])

    return out

if __name__=="__main__":
    run_all()
