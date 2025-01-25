import numpy as np
import matplotlib.pyplot as plt

""""

O modelo KT é um mapa bidimensional. O potencial de membrana é modelado por uma tangente hiperbólica. Com apenas três parâmetros e uma corrente externa, pode 
apresentar uma grande quantidade de comportamentos neuronais. No entando, para obter um bursting neuronal e estimulação cardiaca, o modelo KT foi apliado. 
Isso foi feito adicionando uma terceira e lenta variável z, criando assim o modelo tridimensional KTz.

O modelo KTz logístico utiliza a função logística como aproximação da tangente hiperbólica para poupar custo computacional, uma vez que opera apenas calculando 
frações algébricas ao invés de necessitar expandir em série a função tangente hiperbólica para cada passo. Ainda, como vantangem extra, a análise de estabilidade
linear e de todos os pontos fixos torna-se analítica.

As funções usadas em ambas as versões dos modelos KT e KTz são funções contínuas, de modo que a redefinição do potencil de ação não seja forçada no sistema. Como 
consequência, os picos do KTz têm suas próprias escalas de tempo de subida e descida, o que é útil. Essas escalas de tempo intrínsecas possibilitam o aparecimento 
de picos cardíacos. O uso de modelos simples da família KTZ pode facilitar a compreensão matemática dos fenômenos como explosão induzida por oscilações sublimiares, arritmia cardíaca ou despolarização estudada 
in vitro e in vivo, sendo mais vantajoso que modelos de neurônios baseados em condutância, que possuem enormes espaços de parâmetros. 

"""

#definindo a função logística
def f(u):
    return u / (1. + abs(u))

#definindo o modelo KTz logístico:
def KTz_model(x, y, z, k, T, H, I, l, d, x_R):
    x_next = f((x - k * y + z + H + I)/T)
    y_next = x
    z_next = (1. - d) * z - l * (x - x_R)
    return x_next, y_next, z_next

\
"""

x --> potencial de membrana (em unidades arbitrárias) da célula no momento t
x_next --> potencial da membrana no instante seguinte (t+1)
y --> variável de recuperação no instante t
y_next --> variável de recuperação no instante seguinte (t+1)
z --> canal de corrente iônica lenta no instante t que pode gerar rajadas e picos cardíacos
z_next --> canal de corrente iônica lenta no instante seguinte (t+1)
I --> potencial arbitrário gerado por correntes externas devido a sinapses ou eletrodos
K, T --> parâmetros de controle da dinâmica rápida
delta --> parâmetro de controle que ajusta o tempo de recuperação de z e controla o período refratário
labda, x_R --> ajustam a dinâmica lenta de pico e explosão. Particularmente, labda controla o amortecimento das oscilações, enquanto x_R controla a duração do 
               estouro.
H --> polariza o potencial da membrana 

"""

#condições inicias:
x_0 = -0.35
y_0 = -0.1
z_0 = 0.

#parâmetros:
k = 0.6
T = 0.241
H = 0.
I = 0.
l = 0.001
d = 0.001
x_R = -0.2

#define o número de passos:
n = 1000

#cria uma lista de valores para evoluir x, #y e #z no tempo:
x_values = np.zeros(n)
y_values = np.zeros(n)
z_values = np.zeros(n)

x_inicial, y_inicial, z_inicial = x_0, y_0, z_0

#evolui as funções no tempo:
for i in range(n):
    x_values[i] = x_inicial
    y_values[i] = y_inicial
    z_values[i] = z_inicial
    x_prox, y_prox, z_prox = KTz_model(x_inicial, y_inicial, z_inicial,k, T, H, I, l, d, x_R)
    x_inicial, y_inicial, z_inicial = x_prox, y_prox, z_prox

#para plotar x(t), y(t) e z(t) para cada passo:
plt.figure(figsize=(10,6))
plt.plot(range(n), x_values, label='x', color = 'black')
#plt.plot(range(n),y_values, label='y')
#plt.plot(range(n),z_values,label='z')
plt.xlabel('$n$')
plt.ylabel('$x_n$')
plt.grid(True)
plt.show()