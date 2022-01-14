# Algoritmo de Biseccion
# Ricardo Villagrana Banuelos


# [a,b] se escogen de la gráfica de la función
# error = tolerancia

import numpy as np
import matplotlib.pyplot as plt
from time import time 
import humanfriendly
import datetime


# INGRESO
#fx = lambda x: x**3 + 4*x**2 - 10 
fx = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
dx=0.01
dfx = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)

a = 40 #1
b = 90 #2
tolera = 0.0001

# PROCEDIMIENTO
tramo = b-a
i=0
while not(tramo<tolera):
    start_time = time()
    i=i+1
    c = (a+b)/2
    fa = dfx(a)
    fb = dfx(b)
    fc = dfx(c)
    cambia = np.sign(fa)*np.sign(fc)
    if cambia < 0: 
        a = a
        b = c
    if cambia > 0:
        a = c
        b = b
    tramo = b-a
print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(a,fx(a),i))
end=time()
#print("Time: {:.3f}".format(end - start_time))
print("Time: ",end - start_time)
elapsed_time = time() - start_time 
#elapsed_time = elapsed_time*1000000000000000000
print("Tiempo transcurrido: %f ms." % elapsed_time )
#elapsed_time = time() - start_time
#print("Elapsed time: %.10f seconds." % elapsed_time)


prueba=humanfriendly.format_timespan(elapsed_time)
print("Tiempo: ",prueba)

print("\n")

#my_time = (datetime.date.fromordinal(1) + datetime.timedelta(seconds=elapsed_time)).time()
#print("Tiempo: ",my_time)
#from datetime import datetime, timedelta
#sec = timedelta(elapsed_time)
#time = str(sec)
#print("--> ",time)


# SALIDA
u=fx(c)
h=dfx(c)
print('Raiz en: ', c)
print("\n")

print("Valor sustituido: ",u)
print("\n")

print("Valor 2: ",h)

print('Error en tramo: ', tramo)

