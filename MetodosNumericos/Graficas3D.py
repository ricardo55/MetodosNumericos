import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np


'''
fig=plt.figure()
ax=Axes3D(fig)

x=np.linspace(-4,4,50)
y=np.linspace(-4,4,50)
'''
#def z(x,y):
#	return 

# Dibujar superficie 3D
fig = plt.figure()
 
#axes3d = Axes3D(fig)
axes3d=plt.axes(projection="3d")

def U(x1,x2):
	return 100*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 +   90*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 - (20*x1 + 40*x2)

x = np.linspace(-1,0.1,100)
y = np.linspace(-1,0.1,100)

f = lambda x1,x2:100*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 +   90*(np.sqrt(x1**2  + (x2+1)**2) -1)**2 - (20*x1 + 40*x2);

#[x1,x2]=np.meshgrid(-1:0.1:1,-1:0.1:1);

X,Y = np.meshgrid(x,y)
#Z = np.sqrt(X**2+Y**2)
Z=U(X,Y)

#Wireframe

#axes3d.plot_wireframe(X,Y,Z)
axes3d.plot_surface(X,Y,Z,cmap="viridis")
 


#axes3d.plot_trisurf(X,Y,Z)
#axes3d.plot(x,y,f(x,y))
#axes3d.plot_surface(x,y,f(x,y))
plt.show()




 


