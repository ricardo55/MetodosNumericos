dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
f = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)

def secante(f,a,b,N):
    
    if f(a)*f(b) >= 0:
        print("Fallo el metodo de la secante")
        return None
    p = a
    q = b

    for n in range(1,N+1):

        pParcial = p - f(p)*(q - p)/(f(q) - f(p))
        f_pParcial = f(pParcial)

        if f(p)*f_pParcial < 0:
            p = p
            q = pParcial
        elif f(q)*f_pParcial < 0:
            p = pParcial
            q = q
        elif f_pParcial == 0:
            print("Solucion exacta")
            return pParcial
        else:
            print("Fallo el metetodd secante")
            return None
    return p - f(p)*(q - p)/(f(q) - f(p))


approx=secante(f,40,90,20)
print(approx)
print("\n")
print(fx(approx))