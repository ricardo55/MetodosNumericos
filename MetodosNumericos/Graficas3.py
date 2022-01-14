import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,1,0.2)
y1 = ((10*np.cos(5*x))+(9*x*x)-(4*x)+3)
y2 = (-50*np.sin(5*x)+(18*x)-4)

plt.plot(x,y1,'o',linewidth=3,color=(0.2,0.1,0.4))
plt.hold(True)
plt.plot(x,y2,'-',linewidth=2,color='g')
plt.grid()
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('grafica')
plt.show()