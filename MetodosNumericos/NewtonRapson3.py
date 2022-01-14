# Método de Newton-Raphson
# Ricardo Villagrana Banuelos

import numpy as np
from time import time 

# INGRESO
#fx  = lambda x: x**3 + 4*(x**2) - 10
#dfx = lambda x: 3*(x**2) + 8*x
#start_time = time()
dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
dfx = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)
#dfx = lambda x: (((91682.75*(x**2))-(651310*x)-(242306900))/(((x-20)**2)*((x-165)**2)))
#d2fx = lambda x: (((183365.5*(x**3))-(1953930*(x**2))-(1453841400*x)+(91802876000))/(((x-20)**3)*((x-165)**3)))
d2fx = lambda x: (fx(x+dx)-(2*fx(x))+fx(x-dx))/(dx**2)


x0 = 90
tolera = 0.0001

# PROCEDIMIENTO
tabla = []
tramo = abs(2*tolera)
xi = x0
i=0
while (tramo>=tolera):
    start_time = time()
    i=i+1

    xnuevo = xi - (dfx(xi)/d2fx(xi))
    tramo  = abs(xnuevo-xi)
    tabla.append([xi,xnuevo,tramo])
    xi = xnuevo

end=time()    
print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(xi,fx(xi),i))

#print("Time: {:.3f}".format(end - start_time))
print("Time: ",end - start_time)
elapsed_time = time() - start_time 
#elapsed_time = elapsed_time*1000000000000000000
print("Tiempo transcurrido: %f ms." % elapsed_time )

# convierte la lista a un arreglo.
tabla = np.array(tabla)
n = len(tabla)

# SALIDA
print([' Xi ', ' Xnuevo ', ' Tramo '])
np.set_printoptions(precision = 4)
print(tabla)

print("\n")
print('Raiz en: ', xi)
print('Error de: ',tramo)

#print("\n")

sustituir=fx(xi)
print("Valor sustituido: ",sustituir)
print("\n")

sustituir2=dfx(xi)
print("Valor sustituido 2: ",sustituir2)
print("\n")


#import scipy.optimize as opt
#a= opt.newton(dfx,x0, fprime=d2fx, tol = tolera)
#print(a)

#print("\n")
#ejemplo= opt.newton(fx,x0, fprime=dfx, tol = tolera)
#print(ejemplo)
