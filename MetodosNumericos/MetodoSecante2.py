from time import time 

dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
f = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)
 
def secant(x1, x2, E):

    start_time = time()
    n = 0; xm = 0; x0 = 0; c = 0;
    if (f(x1) * f(x2) < 0):
        while True:
             
            # calculate the intermediate value
            x0 = ((x1 * f(x2) - x2 * f(x1)) /
                            (f(x2) - f(x1)));
 
            # check if x0 is root of
            # equation or not
            c = f(x1) * f(x0);
 
            # update the value of interval
            x1 = x2;
            x2 = x0;
 
            # update number of iteration
            n += 1;
 
            # if x0 is the root of equation
            # then break the loop
            if (c == 0):
                break;
            xm = ((x1 * f(x2) - x2 * f(x1)) /
                            (f(x2) - f(x1)));
             
            if(abs(xm - x0) < E):
                break;
         
        print("Root of the given equation =",
                               round(x0, 6));
        print("No. of iterations = ", n);
        print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(x0,fx(x0),n))

        #print("Time: {:.3f}".format(end - start_time))
        #print("Time: ",end - start_time)
        elapsed_time = time() - start_time 
        #elapsed_time = elapsed_time*1000000000000000000
        print("Tiempo transcurrido: %f ms.\n" % elapsed_time )
         
    else:
        print("Can not find a root in ",
                   "the given inteval");
 
# Driver code
 
# initializing the values
x1 = 40;
x2 = 90;
E = 0.0001;
secant(x1, x2, E);