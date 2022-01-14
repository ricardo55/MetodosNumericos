# ejemplo de trazado de la búsqueda de Adam en un gráfico de contorno de la función de prueba
# Algoritmo de Optimizacion Adam

from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# 3d plot of the test function
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

# definir la función objetivo
def objective(x, y):
	return x**2.0 + y**2.0


# definir rango para entrada
r_min, r_max = -1.0, 1.0
# rango de entrada de muestra uniformemente en incrementos de 0.1
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
#crear una malla desde el eje
x, y = meshgrid(xaxis, yaxis)
# calcular objetivos
results = objective(x, y)
# crea un gráfico de superficie con el esquema de color jet
figure = pyplot.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# mostrar grafica
pyplot.show()




# derivada de la función objetivo
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# algoritmo de descenso de gradiente con adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
	solutions = list()
	# generar un punto inicial
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# inicializar el primer y segundo momento
	m = [0.0 for _ in range(bounds.shape[0])]
	v = [0.0 for _ in range(bounds.shape[0])]

	# ejecutar las actualizaciones de descenso de gradiente
	for t in range(n_iter):
		# calcular gradiente g (t)
		g = derivative(x[0], x[1])
		# construir una solución una variable a la vez
		for i in range(bounds.shape[0]):

			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
			# mhat(t) = m(t) / (1 - beta1(t))
			mhat = m[i] / (1.0 - beta1**(t+1))
			# vhat(t) = v(t) / (1 - beta2(t))
			vhat = v[i] / (1.0 - beta2**(t+1))
			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
			x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)

		# evaluar el punto candidato
		score = objective(x[0], x[1])
		# realizar un seguimiento de las soluciones
		solutions.append(x.copy())
		# informe de progreso
		print('>%d f(%s) = %.5f' % (t, x, score))
	return solutions

# semilla de generador de números pseudoaleatorios
seed(1)
# definir rango para entrada
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# definir las iteraciones totales
n_iter = 60
# tamaño de pasos
alpha = 0.02
# factor de gradiente promedio
beta1 = 0.9
# factor de gradiente cuadrático medio
beta2 = 0.999
# realizar la búsqueda de descenso de gradiente con adam
solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
# rango de entrada de muestra uniformemente en incrementos de 0.1
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)

# crea una malla a partir del eje
x, y = meshgrid(xaxis, yaxis)
# calcular objetivos
results = objective(x, y)
# cree un gráfico de contorno relleno con 50 niveles y esquema de color jet
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# trazar la muestra como círculos negros
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# muestra grafica
pyplot.show()



