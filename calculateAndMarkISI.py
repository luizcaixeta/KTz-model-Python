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

#função para calcular ISI
def calculate_isi(time, potential):
    crossings = []
    for i in range(1, len(potential)):
        if potential[i-1] <= 0 and potential[i] > 0: #quando um potencial é maior que o anterior e positivo, temos um spike
            crossings.append(i)
    crossings = np.array(crossings)
    isis = np.diff(time[crossings]) #calcula o intervalo de tempo entre spikes
    return isis, crossings

#Condições inicias:
x_0 = -0.35
y_0 = -0.1
z_0 = 0.

#Parâmetros:
k = 0.6
T = 0.241
H = 0.
I = 0.
l = 0.001
d = 0.001
x_R = -0.2

#Define o número de passos:
n = 1000

#Cria uma lista de valores para evoluir x, #y e #z no tempo:
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

#criar array de tempo para plotagem
time = np.arange(n)

#calcular ISI
isis, crossings = calculate_isi(time, x_values)

#plotar x_values ao longo do tempo
plt.figure(figsize=(10,6))
plt.plot(time, x_values, label='x', color='black')

#plotar pontos de cruzamento
plt.scatter(time[crossings], x_values[crossings], color='red', label='Crossings')

plt.axhline(0, color='orange', linestyle='--', linewidth=2)

plt.xlabel('$n$')
plt.ylabel('$x_n$')
plt.title('Evolução de $x_n$ e pontos de cruzamento')
plt.grid(True)
plt.legend()
plt.show()

#mostrar ISIs calculados
print("Interspike Intervals (ISIs):", isis)