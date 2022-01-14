def Gaussian_distrib_3d(beta,x,y):
H,B,sigma,x0,y0 = beta
exponente = -( (x-x0)**2 +(y-y0)**2) / (2*sigma*sigma)
return H*np.e**(exponente) + B

gradientF = np.zeros([Nvariables,Ndata])
 
for step in range(0,N):
 H,B,sigma,x0,y0 = beta
 # Calculo de algunos parametros que se repiten
 p1 = (meshx-x0)**2 +(meshy-y0)**2
 p2 = 2*sigma*sigma
 p3 = sigma**3
 p4 = sigma*sigma
 exponente = -p1/p2
 
 Fx = np.ravel( H*np.e**(exponente) + B -z )
 Fx = Fx.transpose()
 
 # Calculo del gradiente
 gradientF[0,:] = np.ravel(np.e**(exponente))
 gradientF[1,:] = 1
 gradientF[2,:] = np.ravel((p1/p3)*H*np.e**(exponente))
 gradientF[3,:] = np.ravel(((meshx-x0)/p4) * H * np.e**(exponente))
 gradientF[4,:] = np.ravel(((meshy-y0)/p4) * H * np.e**(exponente))
 
 gradientFTransp = gradientF.transpose()
 
 nabla = np.dot(gradientF,Fx).reshape(Nvariables,1)
 
 #Calculamos la aproximacion de la matriz Hessiana
 Hessian = np.dot(gradientF,gradientFTransp)
 
 #Resolvemos el sistema algebraico
 p = np.linalg.solve(Hessian, -nabla)
 beta += p


 beta = np.array([100,6,1.2,12,10.5]).reshape(Nvariables,1)

z = data #datos que buscamos aproximar.
x = np.arange(24)
y = np.arange(24)
meshx,meshy = np.meshgrid(x,y)