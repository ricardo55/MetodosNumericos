# Método de Ajuste Cubico Polinomial
# Ricardo Villagrana Banuelos


import numpy as np
import math
from time import time 


# INGRESO
#fx  = lambda x: x**3 + 4*(x**2) - 10
#dfx = lambda x: 3*(x**2) + 8*x
#dx=0.01
f  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
fdx = lambda x,dx: (f(x+dx)-f(x-dx))/(2*dx)
#dfx = lambda x: (((91682.75*(x**2))-(651310*x)-(242306900))/(((x-20)**2)*((x-165)**2)))
#d2fx = lambda x: (((183365.5*(x**3))-(1953930*(x**2))-(1453841400*x)+(91802876000))/(((x-20)**3)*((x-165)**3)))


tolerancia = 0.0001
x1 = 40
x2= 90


#U  = lambda x1,x2: (dfx(x2)+W-Z)/(d2fx(x2)-dfx(x1)+(2*W))

#Z =  lambda x1,x2: ((3*((fx(x1))-fx(x2)))/(x2-x1))+(dfx(x1)+dfx(x2))

#W=  lambda x1,x2: ((x2-x1)/(abs(x2-x1)))+(sqrt((Z**2)-dfx(x1)*dfx(x2)))

'''

if (abs(dfx(x0)<tolerancia)):
    print("x--> \n",x0)
    print("f(x)--> \n",fx(x0))
if (dfx()):
    pass
'''
def ajusteCubicoPolinomial(f,fdx,a,b,iteraciones,error):
    start_time = time()
    dx = 0.01
    x_min = 0
    cont=0

    for i in range(1, iteraciones):

        cont+=1;

        fda = fdx(a,dx)
        fdb = fdx(b,dx)

        z = 3 * (f(a) - f(b)) / (b - a) + fda + fdb
        w = ((b - a) / np.abs(b - a)) * np.sqrt(z * z - fda * fdb)
        miu = calculaMiu(fda,fdb,w,z)

        #print('Iteraciones ' + str(i) + '  a:' + str(a) + ' b:' + str(b))

        if miu>0 and miu<1:
            x_min = b - miu * (b - a)
        elif miu<0:
            x_min = b
        else:
            x_min = a

        if(abs(fdx(x_min,dx))<error):
            break

        else:
            if(fda*fdx(x_min,dx)<0):
                b = x_min
            else:
                a = x_min

    print("x* =", str(x_min) + "  f(x*)=" + str(f(x_min)))
    print("Iteraciones: ",cont)
    print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(x_min,f(x_min),cont))

    #print("Time: {:.3f}".format(end - start_time))
    #print("Time: ",end - start_time)
    elapsed_time = time() - start_time 
    #elapsed_time = elapsed_time*1000000000000000000
    print("Tiempo transcurrido: %f ms.\n" % elapsed_time )
    return x_min




def calculaMiu(fda,fdb,w,z):
    miu = (fdb + w - z) / (fdb - fda + 2 * w)
    return  miu





error = math.exp(1) ** -3
#time 
xmin = ajusteCubicoPolinomial(f,fdx, 40, 90,100,error)
print("x*: " + str(xmin))
print("f(x*):"  + str(f(xmin)))
#print("f(x'): " + str(fdx(xmin, 0.0001)))


