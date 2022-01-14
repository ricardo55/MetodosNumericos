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

# Ricardo Villagrana Banuelos
# DFP



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

# Metodo para la seccion dorada y la busqueda de raices
def busquedaSeccionDorada(x, dire,xi,err):
    a = xi[0]
    b = xi[1]
    tau = 0.381967
    error = 1e-3
    err=error

    alpha1 = a*(1 - tau) + b*tau
    alpha2 = a*tau + b*(1 - tau)
    Ualpha1 = f(x + alpha1*dire)
    Ualpha2 = f(x + alpha2*dire)


    for i in range(100):
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

        if abs(f(x + alpha1*dire) - f(x + alpha2*dire)) < err:
            break;
    return alpha1, Ualpha1



dx=1e-3
tolerancia=1e-3
xi = [-1,1]
x = xi
#Inicializo la matriz hessiana
A=np.eye(len(x))
fx_prev=f(x)


print("Valor inicial de la funcion = %f \n " % fx_prev)


# Aqui inicia el ciclo para realizar las formulas

for j in range(2):
    if j==0:
        direccionPrevia=gradiente(x,dx)
        si_prev = -direccionPrevia
        alpha,fx_prev = busquedaSeccionDorada(x,si_prev,xi,tolerancia);
        
        if LA.norm(direccionPrevia)<tolerancia:
            break
        xActual = x +  alpha*si_prev
        print("J ==0")
        print("----------------------------------")
        print("X[0] --> ",xActual[0])
        print("\n ")
        print("X[1] -->",xActual[1])
        print("\n ")
        print("f(x) -->",fx_prev)
        print("\n ")
        print("Derivada --> ",LA.norm(direccionPrevia))
        print("----------------------------------")


    else:

        valorDelta=xActual-x
        direccionActual=gradiente(xActual,dx)
        valorDeltaDiferencia=direccionActual-direccionPrevia


        primerParte= np.matmul(np.atleast_2d(valorDelta).transpose(),np.atleast_2d(valorDelta) ) /    np.matmul(valorDelta,valorDeltaDiferencia.transpose() )
        segundaParte= np.matmul(np.matmul(np.matmul(A, np.atleast_2d(valorDeltaDiferencia).transpose()), np.atleast_2d(valorDeltaDiferencia)),A) / np.matmul(np.matmul(np.atleast_2d(valorDeltaDiferencia),A ), np.atleast_2d(valorDeltaDiferencia).transpose())
        #print(primerParte)
        #print(segundaParte)


        A = A + primerParte - segundaParte;
        si=np.matmul(-A,direccionActual.transpose())
        si=np.ndarray.flatten(si.transpose())
        alpha,fx_curr = busquedaSeccionDorada(xActual[:],si,xi,tolerancia)
        
        if abs(fx_curr-fx_prev)<tolerancia or LA.norm(direccionActual)<tolerancia:
            break

        fx_prev=fx_curr
        direccionPrevia=direccionActual

        x=xActual
        xActual = x +  alpha*si



        print("Segunda sino")
        print("----------------------------------")
        print("X[0] --> ",xActual[0])
        print("\n ")
        print("X[1] -->",xActual[1])
        print("\n ")
        print("f(x) -->",fx_curr)
        print("\n ")
        print("Derivada --> ",LA.norm(direccionActual))
        print("----------------------------------")



