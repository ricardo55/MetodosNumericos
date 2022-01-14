from time import time 

def newton(f,Df,inicio,termina,max_iter):
    '''Metodo de Newton-Raphson

    Parametros
    ----------
    f : funcion
    Df : function Derivada of f(x).
    inicio : numero inicial
    termina :numero que termina, abs(f(x)) < termina.
    max_iter : numero maximo de iteraciones

    '''
    start_time = time()
    xn = inicio
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < termina:
            print('Solucion encontrada despues de: ',n,'iteraciones.')
            elapsed_time = time() - start_time 
            print("Tiempo transcurrido: %f ms." % elapsed_time )
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Derivada cero, no se encontro solucion')
            elapsed_time = time() - start_time 
            print("Tiempo transcurrido: %f ms." % elapsed_time )
            return None
        xn = xn - fxn/Dfxn
    print('Se exedio el maximo de iteraciones y no hubo solucion')
    return None



#f = lambda x: x**2 - x - 1
#Df = lambda x: 2*x - 1
#newton(f,Df,1,1e-3,10)
dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
dfx = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)
#dfx = lambda x: (((91682.75*(x**2))-(651310*x)-(242306900))/(((x-20)**2)*((x-165)**2)))
#d2fx = lambda x: (((183365.5*(x**3))-(1953930*(x**2))-(1453841400*x)+(91802876000))/(((x-20)**3)*((x-165)**3)))
d2fx = lambda x: (fx(x+dx)-(2*fx(x))+fx(x-dx))/(dx**2)

#p = lambda x: x**3 - x**2 - 1
#Dp = lambda x: 3*x**2 - 2*x
#approx = newton(dfx,d2fx,90,1e-3,15)
approx = newton(dfx,d2fx,40,1e-3,15)
print(approx)

print("\n")

sustituir=fx(approx)
print(sustituir)





