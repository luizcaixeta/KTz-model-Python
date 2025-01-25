import numpy as np
import matplotlib.pyplot as plt

#definindo a função logística
def f(u):
    return u / (1. + abs(u))

#definindo o modelo KTz logístico:
def KTz_model(x, y, z, k, T, H, I, l, d, x_R):
    x_next = f((x - k * y + z + H + I) / T)
    y_next = x
    z_next = (1. - d) * z - l * (x - x_R)
    return x_next, y_next, z_next

#condições inicias:
x_0 = -0.35
y_0 = -0.35
z_0 = 0.

#parâmetros:
k = 0.6
H = 0.
I = 0.
l = 0.001
d = 0.001
x_R = -0.2

#define o número de passos:
n = 1000

#cria uma lista de valores para T
T_values = np.linspace(1., 3., n)

#cria listas para evoluir x, y e z no tempo:
x_values = np.zeros(n)
y_values = np.zeros(n)
z_values = np.zeros(n)

x_inicial, y_inicial, z_inicial = x_0, y_0, z_0

#evolui as funções no tempo:
for i in range(n):
    x_values[i] = x_inicial
    y_values[i] = y_inicial
    z_values[i] = z_inicial
    T_atual = T_values[i]
    x_prox, y_prox, z_prox = KTz_model(x_inicial, y_inicial, z_inicial, k, T_atual, H, I, l, d, x_R)
    x_inicial, y_inicial, z_inicial = x_prox, y_prox, z_prox

#para plotar x(T)
plt.figure(figsize=(10, 6))
plt.plot(T_values, x_values, label='x', color='black')
plt.xlabel('$T$')
plt.ylabel('$x$')
plt.title('')
plt.grid(True)
plt.legend()
plt.show()
