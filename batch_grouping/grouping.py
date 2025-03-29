import pandas as pd
import hashlib

def hash_md5(input_string):
    input_text = str(input_string)
    return hashlib.md5(input_text.encode()).hexdigest()

# Carregar os datasets
estoque = pd.read_excel('database/estoque.xlsx')
lote_max = pd.read_excel('database/lotes.xlsx')

# Criar um dicionário para mapear os lotes máximos por material
lotes = {lote['Material']: lote['Lote Max'] for lote in lote_max.to_dict(orient="records")}

# Criar ID combinando Material e Qualidade
estoque['ID'] = estoque['Material'].astype(str) + " " + estoque['Qualidade'].astype(str)

# Mapear "Lote Max" a partir do dicionário lotes
estoque['Lote Max'] = estoque['Material'].map(lotes)  

# Filtrar apenas os estoques onde "Qtd" seja menor que "Lote Max"
estoque_filtrado = estoque[estoque["Qtd"] < estoque["Lote Max"]]

# Calcular a soma da quantidade por ID
sum_id = estoque_filtrado.groupby('ID', as_index=False)['Qtd'].sum().rename(columns={'Qtd': 'Soma_ID'})

# Adicionar a soma ao DataFrame filtrado
estoque_filtrado = estoque_filtrado.merge(sum_id, on='ID', how='left')

# Contar quantas ocorrências cada ID tem
count_id = estoque_filtrado.groupby('ID', as_index=False)['Qtd'].count().rename(columns={'Qtd': 'Count_ID'})

# Adicionar a contagem ao DataFrame filtrado
estoque_filtrado = estoque_filtrado.merge(count_id, on='ID', how='left')

# Calculando se existe a possibilidade de agrupamento
estoque_filtrado['Coeficiente'] = estoque_filtrado["Soma_ID"] / estoque_filtrado["Lote Max"]
estoque_filtrado['Has Grouping'] = (estoque_filtrado["Count_ID"] >= estoque_filtrado["Coeficiente"]) & (estoque_filtrado["Count_ID"] > 1)

# Salvar Dataset Trabalhado antes da Filtragem
estoque_filtrado.to_excel('new_datasets/lotes_para_agrupamento.xlsx', index=False)

# Filtrar apenas os itens que podem ser agrupados
estoque_filtrado = estoque_filtrado[estoque_filtrado['Has Grouping']]

# Criar listas para armazenar os novos lotes e os dados originais dos lotes utilizados
lotes_agrupados = []
lotes_origem = []

material_old = None
qualidade_old = None
resto = 0
lote_id = 1  # Contador para identificar os novos lotes agrupados

for index, row in estoque_filtrado.sort_values(['Material', 'Qualidade']).iterrows():
    material = row['Material']
    lote = row['Lote']
    quantidade = row['Qtd']
    qualidade = row['Qualidade']
    status = row['Status']
    localizacao = row['Localização']
    quantidade_lote = row['Lote Max']

    if material != material_old or qualidade != qualidade_old:
        # Criar um novo lote agrupado
        new_lote = {
            'Lote ID': hash_md5(lote_id),
            'Material': material,
            'Qualidade': qualidade,
            'Qtd Agrupada': min(quantidade, quantidade_lote),  # Garante que não ultrapasse o lote máximo
            'Status': status
        }
        lotes_agrupados.append(new_lote)
        lote_id += 1  # Incrementa o ID para o próximo lote agrupado
    else:
        # Adiciona a quantidade ao último lote criado
        restante = quantidade_lote - lotes_agrupados[-1]['Qtd Agrupada']
        if quantidade > restante:
            # Se a quantidade ultrapassa o lote máximo, cria um novo lote
            lotes_agrupados[-1]['Qtd Agrupada'] += restante
            new_lote = {
                'Lote ID': hash_md5(lote_id),
                'Material': material,
                'Qualidade': qualidade,
                'Qtd Agrupada': quantidade - restante,
                'Status': status
            }
            lotes_agrupados.append(new_lote)
            lote_id += 1
        else:
            # Caso contrário, apenas soma ao lote existente
            lotes_agrupados[-1]['Qtd Agrupada'] += quantidade

    # Criar dataset de origem dos lotes
    lotes_origem.append({
        'Lote ID': hash_md5(lotes_agrupados[-1]['Lote ID']),
        'Material': material,
        'Qualidade': qualidade,
        'Lote Original': lote,
        'Qtd Usada': quantidade,
        'Status': status,
        'Localização': localizacao
    })

    material_old = material
    qualidade_old = qualidade

# Criar DataFrames finais
df_lotes_agrupados = pd.DataFrame(lotes_agrupados)
df_lotes_agrupados.to_excel('new_datasets/lotes_agrupados.xlsx', index=False)

df_lotes_origem = pd.DataFrame(lotes_origem)
df_lotes_origem.to_excel('new_datasets/lotes_agrupados_origem.xlsx', index=False)

# Exibir os primeiros resultados para verificação
print("Lotes Agrupados:")
print(df_lotes_agrupados.head())

print("\nLotes Originais:")
print(df_lotes_origem.head())