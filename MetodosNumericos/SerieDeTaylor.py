import numpy as np
import math
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 200)
#y = np.zeros(len(x))
e_ala_n = np.zeros(len(x))

labels = ['orden 1', 'orden 3', 'orden 5', 'orden 10']

plt.figure(figsize = (10,8))

exponente=0

for n, label in zip(range(4), labels):
    #y = y + ((-1)**n * (x)**(2*n+1)) / np.math.factorial(2*n+1)
    e_ala_n= e_ala_n +(np.exp(exponente)/ math.factorial(n))*(x-exponente)**n

plt.plot(x,e_ala_n, label = label)

plt.plot(x, np.exp(x), 'k', label = 'Analitico')
plt.grid()
plt.title('Serie de Taylor ')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()