from math import *
from pylab import *

def f2(x):
	return cos(x)-pow(x,3)

def f(T):
	return (204165.5/(330-(2*T)))+(10400/(T-20))


def biseccion(a,b,tol):
	m1=a
	m=b
	k=0

	if (f(a)*f(b)>0):
		print("La funcion no cambia de signo")

	while (abs(m1-m)>tol):
		m1=m
		m=(a+b)/2

		if (f(a)*f(m)<0): #cambia de signo en [a,m]
			b=m

		if (f(m)*f(b)<0): #cambia de signo en [m,b]
			a=m
		print("El intervalo es [",a,",",b,"]")
		k=k+1

	print("X es:",k,"= ",m,"es una buena aproximacion")
	#plot(m1,f(m1))
	#xlabel("x valores")
	#ylabel("y valores")
	#title("Grafica")
	#grid(True)
	#show()

ValorA=40
ValorB=90

#biseccion(0,pi,10**(-6))
biseccion(ValorA,ValorB,10**(-3))



