#Metodo Secante
# Ricardo Villagrana Banuelos

from time import time 

dx=0.01
fx  = lambda x: (204165.5/(330-(2*x)))+(10400/(x-20))
dfx = lambda x: (fx(x+dx)-fx(x-dx))/(2*dx)

#x0=40 #q
#x1=90 #p
#tol=0.0001; 
#n=x1-x0

def secante(f,x,y,n,tol):

	start_time = time()
	itera=0
	for k in range(n):
		itera+=1;
		pParcial=x-(((y-x)/(f(y)-f(x)))*(f(x)))
		error=abs((pParcial-y)/pParcial)
		if error<tol:
			break

		x=y
		y=pParcial
		#print(k,pParcial,f(pParcial),error)
	print('T*={0}  U(T*)= {1} iteraciones={2} \n'.format(pParcial,fx(pParcial),itera))

	#print("Time: {:.3f}".format(end - start_time))
	#print("Time: ",end - start_time)
	elapsed_time = time() - start_time 
	#elapsed_time = elapsed_time*1000000000000000000
	print("Tiempo transcurrido: %f ms.\n" % elapsed_time )
	print("Aproximada: ",pParcial)
	print("\n")
	print("Sustutuida: ",fx(pParcial))
	print("\n")
	print("Iteraciones: ",itera)
	print("\n")
	print("Sustutuida 2: ",dfx(pParcial))
	
		
	

aprox2=secante(dfx,40,90,20,1e-3)
print(aprox2)
print("-----\n")

'''
def secante2(f,x,y,tolerancia):
	error=0.1

	n=0
	pParcial=0

	while error>tolerancia:

		pParcial= x-((y-x/(f(y)-f(x)))*(f(x)))
		x=y
		y=pParcial

		error=abs(f(pParcial))

		n=n+1

	print("Solucion: ",pParcial)
	print("# Iteraciones: ",n)

'''

#aprox=secante(dfx,40,90,0.0003)
#print(aprox)



