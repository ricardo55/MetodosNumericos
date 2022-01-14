import matplotlib.pyplot as plt
import numpy as np


def ecuacion1(x):
	return ((10*np.cos(5*x))+(9*x*x)-(4*x)+3)

def ecuacion1_PrimeraDer(x):

	return ((18*x)-(50*np.sin(5*x))-4)

def ecuacion1_SegundaDer(x):
	return ((-250*np.cos(5*x))+18)



def ecuacion2(x):
	return (2*sin(5*x)+(3*pow(x,3))-(2*pow(x,2))+(3*x)-5)

def ecuacion2_PrimeraDer(x):

	return ((9*pow(x,2))-(4*x)+(10*np.cos(5*x))+3)

def ecuacion2_SegundaDer(x):
	return ((18*x)-(50*np.sin(5*x))-4)

def gradiente(x1,x2,x3):
	return (pow(x1,2))+(2*x1+x2)+(3*pow(x2,2))+(4*pow(x3,2))-(5*x2*x3)

#umesh,vmesh=gradiente()

#Numero de datos
N=12
# Empieza
a=0
#Termina
b=1

x=np.linspace(a,b,N)

#N-1 por los intervalos
dx=(b-a)/(N-1)


y=ecuacion1(x)
yPD=ecuacion1_PrimeraDer(x)
ySD=ecuacion1_SegundaDer(x)


#Calcular la derivada
Yanalitica=np.zeros_like(x)

for i in range(N):
	#Si estas en el primer dato, aplicar derivada
	if i==0:

		#Derivada adelantada
		Yanalitica[i]=(y[i+1]-y[i])/dx

		#Ultimo dato de la derivada
	elif i==N-1:
		
		# Derivada atrasada
		Yanalitica[i]=(y[i]-y[i-1])/dx

		#Cualquier otro dato
	else:
		#derivada central
		Yanalitica[i]=(y[i+1]-y[i-1])/(2*dx)



#Calcular la primera derivada
YprimeraDer=np.zeros_like(x)

for i in range(N):
	if i==0:
		#Derivada adelantada
		YprimeraDer[i]=(yPD[i+1]-yPD[i])/dx
	elif i==N-1:
		# Derivada atrasada
		YprimeraDer[i]=(yPD[i]-yPD[i-1])/dx
	else:
		#derivada central
		YprimeraDer[i]=(yPD[i+1]-yPD[i-1])/(2*dx)



#Calcular la segunda derivada
YsegundaDer=np.zeros_like(x)

for i in range(N):
	if i==0:
		#Derivada adelantada
		YsegundaDer[i]=(ySD[i+2]-2*ySD[i+1]+ySD[i])/dx**2
	elif i==N-1:
		# Derivada atrasada
		YsegundaDer[i]=(ySD[i]-2*ySD[i-1]+ySD[i])/dx**2
	else:
		#derivada central
		YsegundaDer[i]=(ySD[i+1]-2*ySD[i]+ySD[i-1])/dx**2


#Imprimir graficas
plt.plot(x,y,'g-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Derivada')
plt.show()


plt.plot(x,YprimeraDer,'g-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Primera Derivada')
plt.show()

plt.plot(x,YsegundaDer,'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Segunda Derivada')
plt.show()

