import numpy as np
import autograd as ad
#from autograd import grad,jacobian

def f(x):
    fx = 100*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 + 90*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 -(20*x[0]+40*x[1])
    return fx

z=np.array([-1,1],dtype=float)
jacobian_func=jacobian(f)
jacobian_func(z)

i=0
error=100
tol=1e-3
maxiter=1000
M=3
N=3

x0=np.array([1,1,1],dtype=float).reshape(N,1)

while (np.any(abs(error)>tol) and i<maxiter):
    evaluarFuncion=np.array([f(x0),])

def newton(f, x0, tol=1e-3, maxiter=50):
    g = autograd.grad(f)
    h = autograd.hessian(f)

    x = x0
    for _ in range(maxiter):
        delta = np.linalg.solve(h(x), -g(x))
        x = x + delta
        if np.linalg.norm(delta) < tol:
            break

    return x




x0 = np.array([-1, 1])

print(newton(funcx, x0))