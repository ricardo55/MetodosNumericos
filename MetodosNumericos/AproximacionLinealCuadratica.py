# Aproximacion Lineal y Cuadratica y la Grafica
#Ricardo Villagrana Banuelos

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

def funcionNormal(x1,x2):
	return 3*x2-x1/x2

def funcionLineal(x1,x2):
	return 5*x2-x1-2 

def funcionCuadratica(x1,x2):
	#(-2*x2**2+7*x2-2*x2+x1*x2-2)
	return ((-2*x2**2)+(7*x2)-(2*x1)+(x1*x2)-2)


# Valores 
x1=2
x2=1.3

#Prueba de las funciones
print(funcionNormal(x1,x2),"\n")
print(funcionLineal(x1,x2),"\n")
print(funcionCuadratica(x1,x2),"\n")


x1 = np.linspace(1.2, 2.8, 100)
x2 = np.linspace(0.1,1.8, 100)
xx1, xx2 = np.meshgrid(x1, x2)
fun=funcionNormal(xx1,xx2)
plano=funcionLineal(xx1,xx2)
cuad=funcionCuadratica(xx1,xx2)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx1, xx2, fun, cmap=plt.cm.RdYlGn,
                       linewidth=0, antialiased=False)
cmap = colors.LinearSegmentedColormap.from_list("", ["black","black","black"])
ax.plot_surface(xx1, xx2, plano, alpha = 0.5, rstride=1, cstride=1, cmap=cmap,linewidth=0.5, antialiased=True, zorder = 0.3)
#ax.plot_surface(xx1, xx2, cuad, alpha = 0.5, rstride=1, cstride=1, cmap=cmap,linewidth=0.5, antialiased=True, zorder = 0.3)
ax.scatter(2, 1 , funcionNormal(2,1),s=50)
ax.set_zlim(-25,10);
plt.show()



