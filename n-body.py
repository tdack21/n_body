import numpy as np
import matplotlib.pyplot as plt

# constantes
G = 6.67408e-11 # N m^2 / kg^2

# datos iniciales
N = 2 # número de cuerpos
m = np.array([3.42566*10**29, 3.22942*10**29]) # masas de los cuerpos, en kg
r = np.array([[-4.16874*10**13, 6.43957*10**13], [4.42206*10**13, -6.83088*10**13]]) # posiciones de los cuerpos, en m
v = np.array([[-501.2054769, 87.50626131], [531.6618186, -92.82368244]]) # velocidades de los cuerpos, en m/s

# tiempo
t0 = 0.0 # tiempo inicial, en s
tf = 259200000000.0 # tiempo final, en s
dt = tf/1000 # paso temporal, en s

# almacenar las trayectorias
r_list = [r*1] # posiciones de los cuerpos en cada paso temporal
t_list = [t0] # tiempo en cada paso temporal

# cálculo de trayectorias
for t in np.arange(t0, tf, dt):
    # vectores de aceleración
    a = np.zeros_like(r) # inicializar aceleraciones en cero
    for i in range(N):
        for j in range(N):
            if i != j:
                r_rel = r[j] - r[i] # vector de posición relativa
                r_rel_mag = np.linalg.norm(r_rel) # magnitud del vector de posición relativa
                a[i] += -G * m[j] * r_rel / r_rel_mag**3 # sumar aceleración debida a cada cuerpo
    # actualizar velocidades y posiciones
    v += a * dt
    r += v * dt
    # guardar trayectorias
    r_list.append(r*1)
    t_list.append(t)

# convertir listas a arreglos
r_list = np.array(r_list)
t_list = np.array(t_list)

# graficar trayectorias
for i in range(N):
    plt.plot(r_list[:, i, 0], r_list[:, i, 1], label='cuerpo {}'.format(i+1))
plt.legend()
plt.show()
