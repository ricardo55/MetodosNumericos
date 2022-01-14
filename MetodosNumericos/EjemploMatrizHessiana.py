from sympy import *
init_printing()

#Inicio poniendo que son variables estas letras
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

#Funcion, aqui la pongo para saber cual es
f = x**2 + 2*y**2 + 3*z**2 - 6*x**2*z + 4*y*x + 5*y*z**2 - 3*x*y*z
print(f)
print("-----\n")

#Derivada con respecto a x
derivada1=diff(f,x)
print(derivada1)
print("-----\n")

#Derivada con respecto a z
derivada2=diff(f,z)
print(derivada2)
print("-----\n")
#Derivada con respecto a y
derivada3=diff(f,y)
print(derivada3)
print("-----\n")
#Todas estas son derivadas parciales las realizo ya con la funcion 
# para mas rapido.


# Del operador gradiente aplicando la formula, se puede ver que:
# Podemos ver al gradiente, intuitivamente, que al operar sobre una función y la 
# “expande” en un vector columna que contienen sus derivadas parciales. 

#Haciendo na matriz de los elementos solamente, muestra
sbl = Matrix([x,y,z])
print(sbl)
print("-----\n")

#Matriz de f es la funcion
f = Matrix([f])
print(f)
print("-----\n")


#G=f.jacobian(sbl).T
#print(G)
#H = G.jacobian(sbl)
#print(H)
#det(H)

# La definicion de Gradiente en la matriz hessiana tiene que:
# * La hessiana es una matriz de puras segundas derivadas
# * Debe ser simetrica


# Saco el gradiente mediante las matrices, usando el jacobiano
def Grad(f, variables):
    return Matrix([f]).jacobian(Matrix(symbols(variables))).T


# Grad, siempre calculara derivadas por 
# columnas

def Hessian(f, variables):
    g = Grad(f,variables)
    return Grad(g.T, variables)


x1 = Symbol('x1')
x2 = Symbol('x2')

f = (x1 - x2)**2+ x1**3
print(f)
print("-----\n")

a=Grad(f,'x1, x2')
b=Hessian(f, 'x1 x2')

print(a)
print("-----\n")
print(b)
print("-----\n")

print("Ejemplo de Tarea: \n")

#x1 = sympy.Symbol('x1')
#x2 = sympy.Symbol('x2')
x3 = Symbol('x3')
#fU = (204165.5)/(330-2*T) + (10400)/(T-20)
funcionX = (pow(x1,2))+(2*x1+x2)+(3*pow(x2,2))+(4*pow(x3,2))-(5*x2*x3)
print(funcionX)
print("-----\n")
c=Grad(funcionX,'x1, x2, x3')
d=Hessian(funcionX, 'x1 x2 x3')

print(c)
print("-----\n")
print(d)
print("-----\n")

print("Nuevo \n")
f2=(3*x2)-(x1/x2)
print(f2)
print("-----\n")
e=Grad(f2,'x1, x2')
f=Hessian(f2, 'x1 x2')


print(e)
print("-----\n")
print(f)
print("-----\n")




