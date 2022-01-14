import math
from time import time 

dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
f = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)

gr = (math.sqrt(5) + 1) / 2



def metodoSeccionDorada(f, a, b, tol=1e-3):
    start_time = time()
    cont=0
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        cont+=1;

        if fx(c) < fx(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    regreso=(b + a) / 2
    print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(regreso,f(regreso),cont))

    #print("Time: {:.3f}".format(end - start_time))
    #print("Time: ",end - start_time)
    elapsed_time = time() - start_time 
    #elapsed_time = elapsed_time*1000000000000000000
    print("Tiempo transcurrido: %f ms.\n" % elapsed_time )

    
    return regreso



aprox=metodoSeccionDorada(fx,40,90)
print(aprox)
print("-------\n")

subsituido=fx(aprox)
print(subsituido)


