## Generar una gráfica de contorno
# Importar algunas  bibliotecas que necesitaremos
# También instalar los paquetes matplotlib y numpy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# definir la función objetivo
def f(x):
    x1 = x[0]
    x2 = x[1]
    obj = 0.2 + x1**2 + x2**2 - 0.1*math.cos(6.0*3.1415*x1) - 0.1*math.cos(6.0*3.1415*x2)
    return obj

# Lugar de inicio
x_start = [0.8, -0.5]

# Variables de diseño en mesh points
i1 = np.arange(-1.0, 1.0, 0.01)
i2 = np.arange(-1.0, 1.0, 0.01)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
for i in range(x1m.shape[0]):
    for j in range(x1m.shape[1]):
        fm[i][j] = 0.2 + x1m[i][j]**2 + x2m[i][j]**2 \
             - 0.1*math.cos(6.0*3.1415*x1m[i][j]) \
             - 0.1*math.cos(6.0*3.1415*x2m[i][j])

# Crear una gráfica de contorno
plt.figure()
# Especificar contour lines
#lines = range(2,52,2)
# Plot contours
CS = plt.contour(x1m, x2m, fm)#,lines)
# Label contours
plt.clabel(CS, inline=1, fontsize=10)
# agregar texto a la grafica
plt.title('Funcion No-Convexa ')
plt.xlabel('x1')
plt.ylabel('x2')

##################################################
# Simulated Annealing
##################################################
# Numero de ciclos
n = 50
# Número de ensayos por ciclo
m = 50
# Número de soluciones aceptadas
na = 0.0
# Probabilidad de aceptar una peor solución al principio
p1 = 0.7
# Probabilidad de aceptar una peor solución al final
p50 = 0.001
# Temperatura inicial
t1 = -1.0/math.log(p1)
# Temperatura final
t50 = -1.0/math.log(p50)
# Reducción fraccionada en cada ciclo
frac = (t50/t1)**(1.0/(n-1.0))
# Inicializar x
x = np.zeros((n+1,2))
x[0] = x_start
xi = np.zeros(2)
xi = x_start
na = na + 1.0
# Los mejores resultados actuales hasta ahora
xc = np.zeros(2)
xc = x[0]
fc = f(xi)
fs = np.zeros(n+1)
fs[0] = fc
# Temperatura actual
t = t1
#  Promedio DeltaE
DeltaE_avg = 0.0
for i in range(n):
    print('Ciclo: ' + str(i) + ' con Temperatura: ' + str(t))
    for j in range(m):
        # Genera nuevos puntos de prueba
        xi[0] = xc[0] + random.random() - 0.5
        xi[1] = xc[1] + random.random() - 0.5
        # Recortar a los límites superior e inferior
        xi[0] = max(min(xi[0],1.0),-1.0)
        xi[1] = max(min(xi[1],1.0),-1.0)
        DeltaE = abs(f(xi)-fc)
        if (f(xi)>fc):
            # Inicialice DeltaE_avg si se encontró una solución peor
            #   en la primera iteración
            if (i==0 and j==0): DeltaE_avg = DeltaE
            # la función objetivo es peor
            # generar probabilidad de aceptación
            p = math.exp(-DeltaE/(DeltaE_avg * t))
            # determinar si aceptar un punto peor
            if (random.random()<p):
                # acepta la peor solución
                accept = True
            else:
                # no aceptar la peor solución
                accept = False
        else:
            # la función objetivo es menor, aceptar automáticamente
            accept = True
        if (accept==True):
            # actualizar la solución actualmente aceptada
            xc[0] = xi[0]
            xc[1] = xi[1]
            fc = f(xc)
            # Incrementar el número de soluciones aceptadas.
            na = na + 1.0
            # actualizar DeltaE_avg
            DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na
    # Registrar los mejores valores de x al final de cada ciclo
    x[i+1][0] = xc[0]
    x[i+1][1] = xc[1]
    fs[i+1] = fc
    # Bajar la temperatura para el próximo ciclo
    t = frac * t

# imprimir la solución 
print('Mejor solucion: ' + str(xc))
print('Mejor objetivo: ' + str(fc))

plt.plot(x[:,0],x[:,1],'y-o')
plt.savefig('contour.png')

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(fs,'r.-')
ax1.legend(['Objectivo'])
ax2 = fig.add_subplot(212)
ax2.plot(x[:,0],'b.-')
ax2.plot(x[:,1],'g--')
ax2.legend(['x1','x2'])

# Save the figure as a PNG
plt.savefig('iterations.png')

plt.show()