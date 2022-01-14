from scipy.misc import derivative
import numpy as np
from numpy import *
from sympy import diff

# Definiendo la función 
#def f(x):
#    return x**3-np.cos(x)

def f(x):
    return (204165.5/(330-(2*x)))+(10400/(x-20))
#def f2(x):
#    return ((91682.75*(x**2))-65)
def f1(x):
    return (((91682.75*(x**2))-(651310*x)-(242306900))/(((x-20)**2)*((x-165)**2)))

def f2(x):
    return (((183365.5*(x**3))-(1953930*(x**2))-(1453841400*x)+(91802876000))/(((x-20)**3)*((x-165)**3)))


inicio =90
c = 1
#error=10
error=0.0001

'''
while error > 1e-3:
     x1 = inicio - f(inicio)/derivative(f,inicio)
     error=abs(x1 - inicio)
     inicio=x1
     print("Iteracion: ",c,"raíz aproximada a: ",inicio)
     c += 1
'''

for iteration in range(1,15):
    #x1 = inicio - f1(inicio)/derivative(f,inicio)
    x1 = inicio - f1(inicio)/f2(inicio)
    inicio=x1
    print("Iteracion: ",c,"raíz aproximada a: ",inicio)
    c += 1
#print(f1(inicio))

print("\n")
print("La raíz aproximada es de:", inicio)