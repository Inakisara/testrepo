from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

v = np.matrix('3 ; 2')
X=np.matrix('3 3;0 10')


#PARTE A VISUALIZAR V
fig, ax = plt.subplots()
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.arrow(0, 0, v[0,0], v[1,0], shape='full', width=0.01, head_width=0.2)
plt.grid()
plt.show()

#PARTE B VISUALIZAR X*V
y=np.dot(X,v)
fig2, ax2 = plt.subplots()
ax2.set_xlim([-1, 17])
ax2.set_ylim([-1,23 ])
ax2.arrow(0, 0, v[0,0], v[1,0], shape='full', width=0.01, head_width=0.2)
ax2.arrow(0, 0, y[0,0], y[1,0], shape='full', width=0.01, head_width=0.2)
plt.grid()
plt.show()
#PARTE C MATRIZ D(100X2) DE DATOS N(0,1)
D=np.random.random((100,2))
fig3= plt.scatter((D[:,0]),(D[:,1]))
plt.title('100 datos desparramados')
plt.grid()
plt.show()

#D1=D*X graficar D1 y D dif colores
D1=np.dot(D,X)
D1=np.array(D1)
fig3=plt.gca()
fig3.scatter((D[:,0]),(D[:,1]),color='b', label='Datos de D')
fig3.scatter((D1[:,0]),(D1[:,1]),color='r', label='Datos de D'+"'")
plt.legend(loc='upper left')
plt.grid()
plt.show()

#Calcular veps y graficarlos con D, D'. INTERPRETAR
VAP,VEP= np.linalg.eig(X)
print (VEP)
VEP=VEP*2 #AUMENTO EL TAMAÑO DE LOS VEPS PARA VERLOS MEJOR, LO Q ME INTERESA ES SU DIRECCION NO SU TAMAÑO
fig4=plt.gca()
fig4.scatter((D[:,0]),(D[:,1]),color='b', label='Datos de D')
fig4.scatter((D1[:,0]),(D1[:,1]),color='r', label='Datos de D'+"'")
fig4.arrow(0, 0, VEP[0,0], VEP[0,1], shape='full', width=0.01, head_width=0.1, color='black')
fig4.arrow(0, 0, VEP[1,0], VEP[1,1], shape='full', width=0.01, head_width=0.1, color='black')
plt.legend(loc='upper left')
plt.grid()
plt.show()