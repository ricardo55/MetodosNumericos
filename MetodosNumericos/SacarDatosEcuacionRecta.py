import numpy as np
import matplotlib.pyplot as plt
  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

x = np.array([1,2,3,4,5]) 
y = np.array([7,14,15,18,19])
n = np.size(x)
  
x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean
  
Sxy = np.sum(x*y)- n*x_mean*y_mean
Sxx = np.sum(x*x)-n*x_mean*x_mean
  
b1 = Sxy/Sxx
b0 = y_mean-b1*x_mean
print('pendiente b1 es', b1)
print('intercept b0 es', b0)
  
plt.scatter(x,y)
plt.xlabel('Variable Independiente X')
plt.ylabel('Variable Dependiente y')