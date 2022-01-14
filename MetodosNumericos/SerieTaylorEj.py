import matplotlib.pyplot as plt
import numpy as np
import math 
orden = 50
x=(numpy.arange(-5,5, 0.01))
sz=len(x)
e_xa=np.zeros([sz, 1])
a=7
for i in range(0,orden+1):
    temp=(np.exp(a)/ math.factorial(i))*(x-a)**i
    temp=temp.reshape((sz, 1))
    e_xa=e_xa+ temp
     

plt.plot(x,e_xa, label="f\'(x)" , color = "red" )
plt.plot(x,np.exp(x), label="f\'(x)" , color = "black" )
plt.ylim(ymax = 10, ymin = -1);
