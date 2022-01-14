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
# BFGS


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
    #error = 1e-3
    error=err

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

        if abs(f(x + alpha1*dire) - f(x + alpha2*dire)) < error:
            break;
    return alpha1, Ualpha1


# Valores iniciales 
delta=1e-3; 
error=1e-3; 
xi = [-1,1];
x = xi;
A=np.eye(len(x))
fx_prev=f(x)




# Punto nuevo estimado por gradiente
gradientePrevio=gradiente(x,delta)
Si = -gradientePrevio;
alpha,fx_prev = busquedaSeccionDorada(x,Si,xi,error)

xActual = x + alpha*Si
xActual = np.reshape(xActual,(2,1))
x = np.reshape(x,(2,1))

gradientePrevio = np.reshape(gradientePrevio,(2,1))
Si = np.reshape(Si,(2,1))


print("Valor inicial -->  %f \n " % fx_prev)


# Aqui se hace el ciclo for para ir recorriendo los diferentes valores en la matriz de rangos de valores
# y asi poder obtener el valor optimo
for j in range(10):

    deltaX=xActual-x
    gradienteActual=gradiente(xActual,delta)
    
    #Ridemensiono el vector
    gradienteActual=np.reshape(gradienteActual,(2,1))

    deltaG=gradienteActual-gradientePrevio

    # En estas realizar las operaciones de la formula proporcionada de BFGS
    paso1= np.matmul(deltaG,deltaG.transpose()) /  ( np.matmul(deltaG.transpose(),deltaX))
    paso2= np.matmul(gradientePrevio,gradientePrevio.transpose())/  ( np.matmul(gradientePrevio.transpose(),Si))


    #print(paso1)
    #print(paso2)

    # En A se almacenan los pasos que son de la formula proporcionada
    A = A + paso1 + paso2

    # Se saca el Si que es el negativo de la inversa de la funcion
    Si=np.matmul(-1*la.inv(A),gradienteActual)

    # Se hace una busqueda de dos valores en la seccion dorada
    alpha,fx_curr = busquedaSeccionDorada(xActual,Si,xi,error);


    # Aqui se vuelven a actualizar los valores ya sacados en las variables
    fx_prev=f(x)
    gradientePrevio=gradienteActual

    x=xActual
    xActual= x +  alpha*Si

    #print(Si)

    #Condicion de rompimiento del metodo
    if abs(fx_curr-fx_prev)<error or la.norm(gradienteActual)<error:
        break; 
 

print("----------------------------------")
print("X[0] --> ",xActual[0])
print("\n ")
print("X[1] -->",xActual[1])
print("\n ")
print("f(x) -->",fx_curr)
print("\n ")
print("Norma Gradiente --> ",LA.norm(gradienteActual))
print("----------------------------------")










