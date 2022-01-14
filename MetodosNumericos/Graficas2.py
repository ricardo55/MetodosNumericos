import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,1,0.1)
#y = x*np.cos(x)
y=((10*np.cos(5*x))+(9*x*x)-(4*x)+3)

y2=(-50*np.sin(5*x)+(18*x)-4)

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Grafica')
plt.show()

plt.plot(x,y2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Grafica')
plt.show()