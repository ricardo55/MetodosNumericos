from sympy import *
import math
import matplotlib.pyplot as plt
import numpy as np

def calcularGradiente():
    #T = sympy.Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    #fU = (204165.5)/(330-2*T) + (10400)/(T-20)
    funcionX = (pow(x1,2))+(2*x1+x2)+(3*pow(x2,2))+(4*pow(x3,2))-(5*x2*x3)
    
    d1fU = diff(funcionX,x1)
    d2fU = diff(funcionX,x2)
    d3fU = diff(funcionX,x3)
    
    print("Funcion normal\n")
    print(funcionX)
    print("Funcion con gradiente\n")
    print(d1fU)
    print("Funcion con gradiente 2\n")
    print(d2fU)
    print("Funcion con gradiente 3\n")
    print(d3fU)
    
    return None

print(calcularGradiente())


#Numero de datos
N=100
# Empieza
a=0
#Termina
b=1

x=np.linspace(a,b,N)

dx=(b-a)/(N-1)

def ecuacion1():
    funcionX = (pow(x1,2))+(2*x1+x2)+(3*pow(x2,2))+(4*pow(x3,2))-(5*x2*x3)
    
    return funcionX

y=ecuacion1(x)

#Calcular la derivada
Yanalitica=np.zeros_like(x)

for i in range(N):
    Yanalitica[i]=(y[i+dx]-y[i-dx])/2*dx
    print(Yanalitica[i])


