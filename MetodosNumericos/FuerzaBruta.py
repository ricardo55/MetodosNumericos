# Fuerza Bruta Optimizacion
# Ricardo Villagrana Banuelos

import numpy as np
import matplotlib.pyplot as plt
from time import time 


dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
f = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)
fdx = lambda x: (f(x+dx) - f(x-dx))/(2*dx)
 
T=np.linspace(40,90,100)

x=fx(T)
Min=[]
iteraciones=0

for i in range(len(x)-1):
    start_time = time()
    iteraciones=i
    iteraciones=iteraciones+1
    if (x[i]<x[i-1] and x[i]<x[i+1]):
        MinU=(x[i])
        MinT=T[i]
print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(MinT,fx(MinT),iteraciones))

#print("Time: {:.3f}".format(end - start_time))
#print("Time: ",end - start_time)
elapsed_time = time() - start_time 
#elapsed_time = elapsed_time*1000000000000000000
print("Tiempo transcurrido: %f ms.\n" % elapsed_time )
        

print("X: ",MinT)
print("Sustituida: ",MinU)
print("f(x'): " + str(fdx(MinT)))




