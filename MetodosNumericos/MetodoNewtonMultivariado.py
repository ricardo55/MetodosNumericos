import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as LA
import autograd as ad


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




delta=1e-3; 
tolerancia=1e-3; 

#xi = valor inicial
#xi = [-1,1];
xi= x0 = np.array([-1., 1.])
x = xi;
N=30

#fx_prev = es la funcion previa
fx_prev=f(x)

print("Valor inicial de f = %f \n " % fx_prev)

for j in range(N):

    fx_prev=f(x)
    direccion=gradiente(x,delta)
    #direccion=ad.grad(f)
    H=matrizHessiana(x,delta)
    #hessiana=ad.matrizHessiana(f)
    #direccion=np.atleast_2d(direccion)
    direccion=np.atleast_2d(direccion)
    si=np.matmul(-LA.inv(H),direccion.transpose())
    #si=np.matmul(-LA.inv(H),dire.transpose())
    x = x + si.transpose()
    x=np.ndarray.flatten(x)
    fx_curr = f(x)


    if (abs(fx_curr-fx_prev)<tolerancia or LA.norm(direccion)<tolerancia):
        #Se rompe el ciclo
        break;

    print("X[0] --> ",x[0])
    print("\n ")
    print("X[1] -->",x[1])
    print("\n ")
    print("f(x) -->",fx_curr)
    print("\n ")
    print("Derivada --> ",LA.norm(direccion))
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
cs = ax.contour(X,Y,Z,20,linewidths=1)
#cs = ax.contour(X,Y,Z,linewidths=1)
#ax.clabel(cs, fontsize=8)
#plt.contourf(X,Y,fx(X,Y),20)
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




