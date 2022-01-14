import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 200)
y = np.zeros(len(x))

labels = ['orden 1', 'orden 3', 'orden 5', 'orden 7']

plt.figure(figsize = (10,8))


for n, label in zip(range(4), labels):
    y = y + ((-1)**n * (x)**(2*n+1)) / np.math.factorial(2*n+1)
    plt.plot(x,y, label = label)

plt.plot(x, np.sin(x), 'k', label = 'Analytic')
plt.grid()
plt.title('Taylo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()