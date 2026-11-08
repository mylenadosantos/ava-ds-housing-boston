import pandas as pd
from pickle import load

#MODULO DE INFERÊNCIA 

# 4 - Receber dados de uma nova residência

nova_residencia_dados = {
    'CRIM': [0.02731], 'ZN': [0.0], 'INDUS': [7.07], 'CHAS': [0], 'NOX': [0.469], 'RM': [6.421], 'AGE': [78.9], 
    'DIS': [4.9671], 'RAD': [2], 'TAX': [242.0],'PTRATIO': [17.8], 'B': [396.9], 'LSTAT': [9.14], 'MEDV': [0.0]
}

nova_residencia_df = pd.DataFrame(nova_residencia_dados)

print('Dados da nova residência')
print(nova_residencia_df)

# 5 - Normalizar os dados da nova residência

normalizador_carregado = load(open('modelo_normalizador_housing.model', 'rb'))
kmeans_carregado = load(open('modelo_kmeans_housing.model', 'rb'))

colunas_quantitativas_inf = list(normalizador_carregado.feature_names_in_)
colunas_categoricas_inf = ['CHAS', 'RAD']

nova_quant = nova_residencia_df[colunas_quantitativas_inf]
nova_cat = nova_residencia_df[colunas_categoricas_inf]

nova_quant_normalizada = normalizador_carregado.transform(nova_quant.values)
nova_quant_normalizada = pd.DataFrame(nova_quant_normalizada, columns=colunas_quantitativas_inf)

nova_cat_normalizada = pd.get_dummies(nova_cat, columns=colunas_categoricas_inf, drop_first=False).astype(int)

dados_novos_finais = pd.concat([nova_quant_normalizada, nova_cat_normalizada], axis=1)
dados_novos_finais = dados_novos_finais.reindex(columns=kmeans_carregado.feature_names_in_, fill_value=0)

print('Dados quantitativos após a normalização (escala de 0 a 1):')
print(nova_quant_normalizada)

# 6 - Inferir a qual grupo essa nova residência pertence

dados_novos_finais = pd.concat([nova_quant_normalizada, nova_cat_normalizada], axis=1)

dados_novos_finais = dados_novos_finais.reindex(columns=kmeans_carregado.feature_names_in_, fill_value=0)

cluster_predito = kmeans_carregado.predict(dados_novos_finais)
print(f'Nova residência no Cluster: {cluster_predito[0]}')

# 7 - Exibir os dados desnormalizados do centroide inferido

centroides_normalizados = kmeans_carregado.cluster_centers_

centroide_cluster_atual = centroides_normalizados[cluster_predito[0]]

qtd_quantitativas = len(colunas_quantitativas_inf)
centroide_quant_norm = centroide_cluster_atual[:qtd_quantitativas].reshape(1, -1)

centroide_quant_original = normalizador_carregado.inverse_transform(centroide_quant_norm)

df_centroide_real = pd.DataFrame(centroide_quant_original, columns=colunas_quantitativas_inf)

print(f'CENTRÓIDE DO CLUSTER {cluster_predito[0]} (VALORES REAIS / DESNORMALIZADOS)')
print(df_centroide_real.to_string(index=False))
