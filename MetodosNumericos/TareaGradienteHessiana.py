import sympy as sym

def GradienteHessiana(cadenaVariables, funcion):

    vars = sym.symbols(cadenaVariables)
    fx = sym.sympify(funcion)

    Hessiana = sym.zeros(len(fx),len(vars))

    for i, fi in enumerate(fx):

        for j, s in enumerate(vars):

            Hessiana[i,j] = sym.diff(fi, s)

    return Hessiana

a=GradienteHessiana('u1 u2', ['2*u1 + 3*u2','2*u1 - 3*u2'])

b=GradienteHessiana('x1 x2 x3',['(x1**2)+(2*x1*x2)+(3*x2**2)+(4*x3**2)-(5*x2*x3)'])

#funcionX = (pow(x1,2))+(2*x1+x2)+(3*pow(x2,2))+(4*pow(x3,2))-(5*x2*x3)

print(a)

print("--------\n")

print(b)