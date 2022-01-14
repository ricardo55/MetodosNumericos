# import the necessary modules
import numpy as np
import matplotlib.pyplot as plt

# allows us to compute factorials 
import math

# For pretty plots
import seaborn as sns
rc={'lines.linewidth': 2, 'axes.labelsize': 14, 'axes.titlesize': 14, \
    'xtick.labelsize' : 14, 'ytick.labelsize' : 14}
sns.set(rc=rc)


# max number of terms we want to expand to
n_terms = 5

# range of x values to look at
x_vals = np.linspace(0,2,100)

# intialize the array of  y_vals
y_vals = np.zeros([n_terms,len(x_vals)])

# loop through the number of terms
for n in range(n_terms):
    
    # special case for 0th order
    if n == 0:
        y_vals[n,:] = np.ones(len(x_vals))
      
    # otherwise nth order is n-1th order plus new term
    else:
        new_term = x_vals**n / math.factorial(n)
        y_vals[n,:] = y_vals[n-1,:] + new_term

# loop through number of terms and plot
for n in range(n_terms):
    label = str(n) + " order"
    plt.plot(x_vals,y_vals[n,:], label=label)
    
# plot the real function  
plt.plot(x_vals, np.exp(x_vals), label="$e^x$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()