import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as LA
import autograd as ad
from time import time 


# Funcion 
def f(x):
    fx = 100*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 + 90*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 -(20*x[0]+40*x[1])
    return fx

#Metodo para obtener el gradiente mediante las aproximadas
def gradiente(x,delta):
    grad=np.zeros(2)
    grad[0]=(f([x[0]+delta,x[1]])- f([x[0]-delta,x[1]]))/(2*delta)
    grad[1]=(f([x[0],x[1]+delta])- f([x[0],x[1]-delta]))/(2*delta)
    return grad

# Metodo para realizar la matriz Hessiana utilizando las formulas aproximadas
def matrizHessiana(x,delta):
    hessiana=np.zeros([2,2])
    hessiana[0,0]= ( f([x[0]+delta,x[1]])  - 2*f(x) + f([x[0]-delta,x[1]]) )/ delta**2;
    hessiana[1,1]= ( f([x[0],x[1]+delta])  - 2*f(x) + f([x[0],x[1]-delta]) )/ delta**2; 
    hessiana[0,1]= ( f([x[0]+delta,x[1]+delta]) - f([x[0]+delta,x[1]-delta]) - f([x[0]-delta,x[1]+delta]) + f([x[0]-delta,x[1]-delta]) )/ (4*(delta**2));
    hessiana[1,0]=hessiana[0,1]
    return hessiana

def busquedaSeccionDorada(x, dire):
    a = -5
    b = 5
    tau = 0.381967
    epsilon = 1e-5

    alpha1 = a*(1 - tau) + b*tau
    alpha2 = a*tau + b*(1 - tau)
    Ualpha1 = f(x + alpha1*dire)
    Ualpha2 = f(x + alpha2*dire)


    for _ in range(0, 1000):
        if Ualpha1 > Ualpha2:

            a = alpha1
            alpha1 = alpha2

            Ualpha1 = Ualpha2
            alpha2 = tau*a + (1 - tau)*b

            Ualpha2 = f(x + alpha2*dire)

        else:
            b = alpha2
            alpha2  = alpha1
            Ualpha2 = Ualpha1

            alpha1  = tau*b + (1 - tau)*a
            Ualpha1 = f(x + alpha1*dire)

        if abs(f(x + alpha1*dire) - f(x + alpha2*dire)) < epsilon:
            break
    return alpha1, Ualpha1


'''
def busquedaSeccionDorada(x,dire):
    start_time = time()
    a = -1
    b = 1
    tau = 0.381967
    tolerencia = 1e-3

    cont = 0
    registro = []

    while(True):
        # Calcular alpha1 y alpha2
        alpha1 = a*(1 - tau) + b*tau
        alpha2 = a*tau + b*(1 - tau)

        # Calcular f(alpha1) y f(alpha2)
        U_alpha1 = f(x + alpha1*dire)
        U_alpha2 = f(x + alpha2*dire)
        
        if(U_alpha1 > U_alpha2):
            a = alpha1
        else:
            b = alpha2       

        cont = cont + 1
        registro.append([cont, alpha1, U_alpha1])
        #print("It: {:02d} - Temp: {:.10f} - Costo: {:.10f}".format(cont, alpha1, U_alpha1))
        #print("f(x'): " + str(fdx(alpha1)))

        if(np.abs(U_alpha1 - U_alpha2) < tolerencia):
            #print("-------------------------------------------------------")
            #print("It: {:02d} - Temp: {:.10f} - Costo: {:.10f}".format(cont, alpha1, U_alpha1))
            #print("f(x'): " + str(fdx(alpha1)))
            break
    #print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(alpha1,fx(alpha1),cont))
    #print("f(x'): " + str(f(alpha1)))

    #print("Time: {:.3f}".format(end - start_time))
    #print("Time: ",end - start_time)
    elapsed_time = time() - start_time 
    #elapsed_time = elapsed_time*1000000000000000000
    #print("Tiempo transcurrido: %f ms.\n" % elapsed_time )
            
    return registro

'''


dx=1e-3 
tole=1e-3
x = [-1,1]

a=-1
b=1
tau=0.381967
tole = 1e-3


f_prev=f(x)
print('Valor de la funcion = {0:.2f} '.format(f_prev))
#print('Iter o.\t x-vector \tf(x)\t Norm ')
#print('---------------------------------------------------')
for i in range(1, 500):

    gradientee=gradiente(x,dx)
    H=matrizHessiana(x,dx) 

    si=np.matmul(-1*la.inv(H),gradientee.transpose())
    si = si.transpose()
    si=np.ndarray.flatten(si)

    alpha,fx_curr = busquedaSeccionDorada(x,si);


    if abs(fx_curr-f_prev)<tole or la.norm(gradientee)<tole:
        break;
    x = x +  alpha*si

    f_prev=fx_curr

    print("X[0] --> ",x[0])
    print("\n ")
    print("X[1] -->",x[1])
    print("\n ")
    print("f(x) -->",fx_curr)
    print("\n ")
    print("Normalizada --> ",la.norm(gradientee))
    print("----------------------------------")


import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np

fig = plt.figure()
#fig, ax = plt.subplots()
ax = fig.add_subplot(111)

x = y = np.linspace(-1, 1, 100)
X,Y = np.meshgrid(x,y)
#Z = (-4*X)/(X**2 + Y**2 + 1)
fx = lambda x1,x2:100*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 +   90*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 - (20*x1 + 40*x2);

Z=fx(X,Y)
#print(Z)

#axes3d = fig.add_subplot(121)
#axes3d=plt.axes(projection="3d")
#axes3d.plot_surface(X,Y,Z,cmap="viridis")
#plt.show()

#cs = ax.contour(X,Y,Z,8, colors = 'black')
plt.plot(x,y)
cs = ax.contour(X,Y,Z,8,linewidths=1)
#cs = ax.contour(X,Y,Z,linewidths=1)
#ax.clabel(cs, fontsize=8)
#plt.contourf(X,Y,fx(X,Y),8)
#plt.scatter(X,Y,c=Z,s=20) 
#plt.xlim(-1,1) 
#plt.ylim(-1,1)
#plt.clabel(cs,inline=True,fontsize=10)

#plt.show()
#plt.plot(fx(-1,1))


#plt.axhline(fx(-1,1), color="black")
ax.set_title('Contour Plot') 
ax.set_xlabel('feature_x') 
ax.set_ylabel('feature_y') 


#ax.clablel(C,inline=1,fontsize=10)
plt.show()









