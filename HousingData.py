import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load
import math

# MÓDULO DE TREINAMENTO

# 1- Normalizar a base com cuidado sobre dados categóricos e dados quantitativos

dados = pd.read_csv('HousingData.csv')

dados = dados.fillna(dados.median())

colunas_categoricas = ['CHAS', 'RAD']
colunas_quantitativas = [col for col in dados.columns if col not in colunas_categoricas]

dados_quant = dados[colunas_quantitativas]
dados_cat = dados[colunas_categoricas]

normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_quant)

dump(modelo_normalizador, open('modelo_normalizador_housing.model', 'wb'))

dados_quant_normalizados = modelo_normalizador.transform(dados_quant)
dados_quant_normalizados = pd.DataFrame(dados_quant_normalizados, columns=dados_quant.columns)

dados_cat_normalizados = pd.get_dummies(dados_cat, columns=colunas_categoricas, drop_first=False).astype(int)

dados_finais = pd.concat([dados_quant_normalizados, dados_cat_normalizados], axis=1)
print('Base tratada e normalizada')
print(dados_finais.head())


# 2- Determinar o número ótimo de grupos com o método do cotovelo (distâncias)

distorcoes = []
K = range(1, 10)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init='auto').fit(dados_finais)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_finais, cluster_model.cluster_centers_, 'euclidean'),
                axis=1
            ) / dados_finais.shape[0]
        )
    )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]

distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0)**2 + (xn - x0)**2)
    distancias.append(numerador / denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print(f'Número ótimo de clusters: {n_clusters_otimo}')

#Plotar gráfico
fig, ax = plt.subplots()
ax.plot(K, distorcoes, marker='o', color='b', linestyle='-')
ax.set(xlabel='Número de Clusters (K)', ylabel='Distorções', title='Método do Cotovelo - Boston Housing')
ax.grid(True)
plt.show()

# 3- Treinar e salvar o modelo

modelo_kmeans_final = KMeans(n_clusters=n_clusters_otimo, random_state=42, n_init='auto')
modelo_kmeans_final.fit(dados_finais)

dump(modelo_kmeans_final, open('modelo_kmeans_housing.model', 'wb'))
