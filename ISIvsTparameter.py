import numpy as np
import matplotlib.pyplot as plt

#definindo a função logística
def f(u):
    return u / (1. + abs(u))

#definindo o modelo KTz logístico
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

#função para simular o modelo KTz logístico com um valor específico de T
def simulate_model(T_value, n=1000):
    #condições iniciais:
    x_0 = 0.35
    y_0 = 0.1
    z_0 = 0.

    #parâmetros fixos:
    k = 0.6
    H = 0.
    I = 0.
    l = 0.001
    d = 0.001
    x_R = -0.2

    #listas para armazenar os resultados
    x_values = np.zeros(n)
    y_values = np.zeros(n)
    z_values = np.zeros(n)

    x_inicial, y_inicial, z_inicial = x_0, y_0, z_0

    for i in range(n):
        x_values[i] = x_inicial
        y_values[i] = y_inicial
        z_values[i] = z_inicial
        x_prox, y_prox, z_prox = KTz_model(x_inicial, y_inicial, z_inicial, k, T_value, H, I, l, d, x_R)
        x_inicial, y_inicial, z_inicial = x_prox, y_prox, z_prox

    #calcular ISI considerando spikes quando x está ficando positivo
    isis, _ = calculate_isi(np.arange(n), x_values)
    
    return isis

#valores de T para variar
T_values = np.linspace(0.1, 0.4, 50)  # Variando de 0.1 a 0.4 com 200 valores

#lista para armazenar os ISIs médios para cada valor de T
isis_means = []

#simular o modelo para cada valor de T e calcular o ISI médio
for T_val in T_values:
    isis = simulate_model(T_val)
    if len(isis) > 0:
        isis_means.append(np.mean(isis))
    else:
        isis_means.append(0)

#plotar ISI médio em função de T
plt.figure(figsize=(10,6))
plt.scatter(T_values, isis_means, color='black')

plt.xlabel('Valor de T')
plt.ylabel('ISI Médio')
plt.title('ISI Médio em função do parâmetro T')
plt.grid(True)
plt.show()