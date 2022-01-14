import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as LA
import autograd as ad
from time import time 
from mpl_toolkits import mplot3d



# Funcion 
def f(x):
    fx = 100*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 + 90*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 -(20*x[0]+40*x[1])
    return fx

#Metodo para obtener el gradiente mediante las aproximadas
def gradiente(x,delta):
    grad=np.zeros(2)
    grad[0]=(f([x[0]+delta,x[1]])- f([x[0]-delta,x[1]]))/(2*delta)
    grad[1]=(f([x[0],x[1]+delta])- f([x[0],x[1]-delta]))/(2*delta)
    return grad

# Metodo para realizar la matriz Hessiana utilizando las formulas aproximadas
def matrizHessiana(x,delta):
    hessiana=np.zeros([2,2])
    hessiana[0,0]= ( f([x[0]+delta,x[1]])  - 2*f(x) + f([x[0]-delta,x[1]]) )/ delta**2;
    hessiana[1,1]= ( f([x[0],x[1]+delta])  - 2*f(x) + f([x[0],x[1]-delta]) )/ delta**2; 
    hessiana[0,1]= ( f([x[0]+delta,x[1]+delta]) - f([x[0]+delta,x[1]-delta]) - f([x[0]-delta,x[1]+delta]) + f([x[0]-delta,x[1]-delta]) )/ (4*(delta**2));
    hessiana[1,0]=hessiana[0,1]
    return hessiana

def busquedaSeccionDorada(x, dire,xi):
    a = xi[0]
    b = xi[1]
    tau = 0.381967
    epsilon = 1e-3

    alpha1 = a*(1 - tau) + b*tau
    alpha2 = a*tau + b*(1 - tau)
    Ualpha1 = f(x + alpha1*dire)
    Ualpha2 = f(x + alpha2*dire)


    for _ in range(0, 100):
        if Ualpha1 > Ualpha2:

            a = alpha1
            alpha1 = alpha2

            Ualpha1 = Ualpha2
            alpha2 = tau*a + (1 - tau)*b

            Ualpha2 = f(x + alpha2*dire)

        else:
            b = alpha2
            alpha2  = alpha1
            Ualpha2 = Ualpha1

            alpha1  = tau*b + (1 - tau)*a
            Ualpha1 = f(x + alpha1*dire)

        if abs(f(x + alpha1*dire) - f(x + alpha2*dire)) < epsilon:
            break;
    return alpha1, Ualpha1




dx=1e-3; 
tolerancia=1e-3; 

xi = np.array([-1., 1.]);
x = xi;



lambdaa = float(1e3);
fx_prev=f(x)


for j in range(30):

    fx_prev=f(x)
    direccion=gradiente(x,dx)
    H=matrizHessiana(x,dx)

    direccion=np.atleast_2d(direccion)
    si=np.matmul(-LA.inv(H+lambdaa*np.eye(len(x))),direccion.transpose())
    
    x = x + si.transpose() 
    x=np.ndarray.flatten(x)
    fx_curr = f(x)

    if fx_curr < fx_prev:
        lambdaa = lambdaa/2
    else:
        lambdaa = 2*lambdaa

    if abs(fx_curr-fx_prev)<tolerancia or LA.norm(direccion)<tolerancia:
        break;


    print("----------------------------------")
    print("X[0] --> ",x[0])
    print("\n ")
    print("X[1] -->",x[1])
    print("\n ")
    print("f(x) -->",fx_curr)
    print("\n ")
    print("Normalizada --> ",LA.norm(direccion))
    print("----------------------------------")


fig = plt.figure()
#fig, ax = plt.subplots()
ax = fig.add_subplot(111)

x = y = np.linspace(-1, 1, 100)
X,Y = np.meshgrid(x,y)
#Z = (-4*X)/(X**2 + Y**2 + 1)
fx = lambda x1,x2:100*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 +   90*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 - (20*x1 + 40*x2);

Z=fx(X,Y)

#plt.plot(x,y)
#cs = ax.contour(X,Y,Z,8,linewidths=1)

plt.contour(X,Y,Z,20)
plt.plot(-1,1, 'ro--', linewidth=2, markersize=6)
plt.plot(0.5,0, 'go--', linewidth=2, markersize=6)
plt.show()

'''
for j in range(30):

    fx_prev=f(x)
    direccion=gradiente(x,dx)
    H=matrizHessiana(x,dx)

    direccion=np.atleast_2d(direccion)
    si=np.matmul(-LA.inv(H+lambdaa*np.eye(len(x))),direccion.transpose())
    
    x = x + si.transpose() 
    x=np.ndarray.flatten(x)
    fx_curr = f(x)

    if fx_curr < fx_prev:
        lambdaa = lambdaa/2
    else:
        lambdaa = 2*lambdaa

    if abs(fx_curr-fx_prev)<tolerancia or LA.norm(direccion)<tolerancia:
        break;
    plt.plot([fx_prev[0],fx_curr[0]],[fx_prev[1],fx_curr[1]], 'k', linewidth=2,)


    print("----------------------------------")
    print("X[0] --> ",x[0])
    print("\n ")
    print("X[1] -->",x[1])
    print("\n ")
    print("f(x) -->",fx_curr)
    print("\n ")
    print("Normalizada --> ",LA.norm(direccion))
    print("----------------------------------")

'''

