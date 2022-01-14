import matplotlib.pyplot as plt
import numpy as np
from time import time 

dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
f = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)


def busquedaSeccionDorada(num1,num2,numTau,tol):
    start_time = time()
    a = num1
    b = num2
    tau = numTau
    tolerencia = tol

    cont = 0
    registro = []

    while(True):
        # Calcular alpha1 y alpha2
        alpha1 = a*(1 - tau) + b*tau
        alpha2 = a*tau + b*(1 - tau)

        # Calcular f(alpha1) y f(alpha2)
        U_alpha1 = fx(alpha1)
        U_alpha2 = fx(alpha2)
        
        if(U_alpha1 > U_alpha2):
            a = alpha1
        else:
            b = alpha2       

        cont = cont + 1
        registro.append([cont, alpha1, U_alpha1])
        print("It: {:02d} - Temp: {:.10f} - Costo: {:.10f}".format(cont, alpha1, U_alpha1))
        #print("f(x'): " + str(fdx(alpha1)))

        if(np.abs(U_alpha1 - U_alpha2) < tolerencia):
            print("-------------------------------------------------------")
            print("It: {:02d} - Temp: {:.10f} - Costo: {:.10f}".format(cont, alpha1, U_alpha1))
            #print("f(x'): " + str(fdx(alpha1)))
            break
    print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(alpha1,fx(alpha1),cont))
    print("f(x'): " + str(f(alpha1)))

    #print("Time: {:.3f}".format(end - start_time))
    #print("Time: ",end - start_time)
    elapsed_time = time() - start_time 
    #elapsed_time = elapsed_time*1000000000000000000
    print("Tiempo transcurrido: %f ms.\n" % elapsed_time )
            
    return registro


a = 40
b = 90
#tau = 2 - 1.618033988
tau=0.381967
tole = 1e-3

busquedaSeccionDorada(a,b,tau,tole)





