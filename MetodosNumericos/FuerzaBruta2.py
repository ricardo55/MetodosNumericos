import numpy as np 
dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
f = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)

def F_U(x):
    fx = 100*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 + 90*(np.sqrt(x[0]**2+(x[1]+1)**2)-1)**2 -(20*x[0]+40*x[1])
    return fx

def F_U2(x1,x2):
    fx = 100*(np.sqrt(x1**2+(x2+1)**2)-1)**2 + 90*(np.sqrt(x1**2+(x2+1)**2)-1)**2 -(20*x1+40*x2)
    return fx


def brute_force(x1,x2,min,x1min,x2min):
    for i in range(1,x1):
        #for j in range(1,x2):
        fu = fx(x1[i])
        if fu<min:
            min = fu 
            x1min = x1[i]
            x2min = x2[i]
    print('Minimum value = {0} ,  x1 ={1}  ,  x2 = {2}'.format(min,x1min,x2min))
    
    
    
    
#x1 = np.arange(-1,1,0.001)
#x2 = np.arange(-1,1,0.001)
x1=40
x2=90
x1min = -1
x2min = -1
min = 1e20
brute_force(x1,x2,min,x1min,x2min)

