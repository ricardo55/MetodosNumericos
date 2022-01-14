import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm

def funcx(x):
    fx = 100*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 + 90*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 -(20*x[0]+40*x[1])
    return fx

def gradient(x,delta):
    grad=np.zeros(2)
    grad[0]=(funcx([x[0]+delta,x[1]])- funcx([x[0]-delta,x[1]]))/(2*delta)
    grad[1]=(funcx([x[0],x[1]+delta])- funcx([x[0],x[1]-delta]))/(2*delta)
    return grad

def golden(x,search,xi,eps):
    a = xi[0]
    b = xi[1]
    tau = 0.381967
    alpha1 = a*(1-tau) + b*tau
    alpha2 = a*tau + b*(1-tau)
    falpha1 = funcx(x+alpha1*search)
    falpha2 = funcx(x+alpha2*search)
    for i in range(100):
        if falpha1 > falpha2:
            a = alpha1
            alpha1 = alpha2
            falpha1 = falpha2
            alpha2 = tau*a + (1-tau)*b
            falpha2 = funcx(x+alpha2*search)
        else:
            b = alpha2
            alpha2 = alpha1
            falpha2 = falpha1
            alpha1 = tau*b + (1-tau)*a
            falpha1 = funcx(x+alpha1*search)

        if np.abs(funcx(x+alpha1*search)- funcx(x+alpha2*search)) < eps :
            break;
    return alpha1,falpha1


delta=1e-3 
eps=1e-3 
xlim = [-1,1]
x = [-1,1]
fx_prev=funcx(x)
print("Initial function value = %f \n " % fx_prev)
print("No. \t \t x-vector \t   f(x) \t Norm_grad \n")
print("______________________________________________________________________\n")
for j in range(30):
    grad=gradient(x,delta)
    si=-grad
    alpha,fx_curr = golden(x,si,xlim,eps)
    x = x + alpha*si
    if abs(fx_curr-fx_prev)<eps or LA.norm(grad)<eps:
        break
    fx_prev=fx_curr
    print("%d\t %f %f\t %f \t%f \n" %(j,x[0],x[1],fx_curr,LA.norm(grad)))


#---------------------------------
# Graficas

delta=1e-3 
eps=1e-3 
xlim = [-1,1]
xp = [-1,1]
fx_prev=funcx(x)
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, y)
z = 100*(np.sqrt(xx**2+(yy+1)**2)-1)**2 + 90*(np.sqrt(xx**2+(yy+1)**2)-1)**2 -(20*xx+40*yy)
plt.contour(x,y,z,20)
plt.plot(-1,1, 'ro--', linewidth=2, markersize=6)
plt.plot(0.5,0, 'go--', linewidth=2, markersize=6)
for j in range(30):
    grad=gradient(xp,delta)
    si=-grad
    alpha,fx_curr = golden(xp,si,xlim,eps)
    xc = xp + alpha*si
    if abs(fx_curr-fx_prev)<eps or LA.norm(grad)<eps:
        break
    fx_prev=fx_curr
    plt.plot([xp[0],xc[0]],[xp[1],xc[1]], 'k', linewidth=2,)
    xp=xc
plt.show()



