import numpy as np
import matplotlib.pyplot as plt

#definindo a função logística
def f(u):
    return u / (1. + abs(u))

#definindo o modelo KTz logístico:
def KTz_model(x, y, z, k, T, H, I, l, d, x_R):
    x_next = f((x - k * y + z + H + I)/T)
    y_next = x
    z_next = (1. - d) * z - l * (x - x_R)
    return x_next, y_next, z_next

#função para calcular ISI considerando spikes quando x está ficando positivo
def calculate_isi(time, potential):
    crossings = []
    for i in range(1, len(potential)):
        if potential[i-1] <= 0 and potential[i] > 0:
            crossings.append(time[i])
    crossings = np.array(crossings)
    isis = np.diff(crossings)
    return isis, crossings

#condições inicias:
x_0 = -0.35
y_0 = 0.
z_0 = 0.

#parâmetros:
k = 0.6
T = 0.249
H = 0.
I = 0.
l = 0.001
d = 0.001
x_R = -0.2

#define o número de passos:
n = 100

#cria uma lista de valores para evoluir x, #y e #z no tempo:
x_values = np.zeros(n)
y_values = np.zeros(n)
z_values = np.zeros(n)

x_inicial, y_inicial, z_inicial = x_0, y_0, z_0

for i in range(n):
    x_values[i] = x_inicial
    y_values[i] = y_inicial
    z_values[i] = z_inicial
    x_prox, y_prox, z_prox = KTz_model(x_inicial, y_inicial, z_inicial,k, T, H, I, l, d, x_R)
    x_inicial, y_inicial, z_inicial = x_prox, y_prox, z_prox

#calcular ISI considerando spikes quando x está ficando positivo
isis, spike_times = calculate_isi(np.arange(n), x_values)

#plotar ISI pelo tempo
plt.figure(figsize=(10,6))
plt.plot(spike_times[:-1], isis, marker='o', linestyle='-', color='black')

plt.xlabel('Tempo')
plt.ylabel('ISI')
plt.title('Interspike Interval (ISI) pelo tempo')
plt.grid(True)
plt.show()
