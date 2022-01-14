#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import pyplot
import math
# Función cuadrática.

def primeraDer(x):
    return ((10*pyplot.os(5*x))+(9*x*x)-(4*x)+3)

def segundaDer(x):
    return (-50*pyplot.sin(5*x)+(18*x)-4)

def f1(x):
    return 2*(x**2) + 5*x - 2
# Función lineal.
def f2(x):
    return 4*x + 1
# Valores del eje X que toma el gráfico.
x = range(-10, 15)
# Graficar ambas funciones.
#pyplot.plot(x, [f1(i) for i in x])
#pyplot.plot(x, [f2(i) for i in x])
pyplot.plot(x, [primeraDer(i) for i in x])
pyplot.plot(x, [segundaDer(i) for i in x])
# Establecer el color de los ejes.
pyplot.axhline(0, color="black")
pyplot.axvline(0, color="black")
# Limitar los valores de los ejes.
pyplot.xlim(-10, 10)
pyplot.ylim(-10, 10)
# Guardar gráfico como imágen PNG.
pyplot.savefig("output.png")
# Mostrarlo.
pyplot.show()