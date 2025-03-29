import pandas as pd

def regra_quantidade(quantidade_atual: int, quantidade_salva: int, quantidade_max: int) -> (bool, int, int):
    """
    Verifica se a quantidade total ultrapassa o máximo permitido e ajusta conforme necessário.
    
    Args:
        quantidade_atual: Quantidade do lote atual sendo processado
        quantidade_salva: Quantidade já acumulada no lote agrupado
        quantidade_max: Quantidade máxima permitida por lote
    
    Returns:
        Tupla contendo:
        - Flag indicando se um novo lote deve ser criado (True) ou não (False)
        - Quantidade a ser adicionada ao lote atual
        - Quantidade restante (se houver) para criar um novo lote
    """
    if quantidade_atual + quantidade_salva > quantidade_max:
        qtd_restante = (quantidade_max - quantidade_salva)
        return True, qtd_restante, quantidade_atual - qtd_restante
    return False, quantidade_atual, 0

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

# Iniciar o processamento por grupo de Material e Qualidade
lote_id = 1  # Contador para identificar os novos lotes agrupados

# Ordenar por Material e Qualidade, depois por quantidade (do MAIOR para o MENOR)
# Isso prioriza agrupar lotes menores a lotes maiores
estoque_ordenado = estoque_filtrado.sort_values(['ID', 'Qtd'], ascending=[True, True])

# Processar por grupos de ID (Material + Qualidade)
for id_grupo, grupo in estoque_ordenado.groupby('ID'):
    quantidade_acumulada = 0
    quantidade_lote_max = grupo['Lote Max'].iloc[0]
    
    # Iniciar um novo lote agrupado
    lote_atual_id = lote_id
    
    # Primeiro, identificar lotes que já estão próximos da capacidade máxima
    # Isso permite que lotes quase cheios sejam priorizados para completar
    lotes_grandes = grupo[grupo['Qtd'] > quantidade_lote_max * 0.7].copy()
    lotes_pequenos = grupo[grupo['Qtd'] <= quantidade_lote_max * 0.7].copy()
    
    # Processar primeiro os lotes grandes (próximos do máximo)
    for index, row in lotes_grandes.iterrows():
        material_id = row['ID']
        material = row['Material']
        lote = row['Lote']
        quantidade = row['Qtd']
        qualidade = row['Qualidade']
        status = row['Status']
        localizacao = row['Localização']
        
        # Verificar se precisa iniciar um novo lote
        if quantidade_acumulada >= quantidade_lote_max or quantidade_acumulada == 0:
            lote_atual_id = lote_id
            quantidade_acumulada = 0
        
        quantidade_restante = quantidade
        while quantidade_restante > 0:
            # Calcular quanto podemos adicionar ao lote atual
            espaco_disponivel = quantidade_lote_max - quantidade_acumulada
            quantidade_adicionar = min(espaco_disponivel, quantidade_restante)
            
            # Registrar a origem do lote
            if quantidade_adicionar > 0:
                lotes_origem.append({
                    'ID': material_id,
                    'Material': material,
                    'Qualidade': qualidade,
                    'Quantidade': quantidade_adicionar,
                    'Lote_id': lote_atual_id,
                    'Localização': localizacao,
                    'Lote': lote,
                    'Status': status
                })
                
                quantidade_acumulada += quantidade_adicionar
                quantidade_restante -= quantidade_adicionar
                
                # Se o lote atual está completo, criar um novo lote
                if quantidade_acumulada >= quantidade_lote_max and quantidade_restante > 0:
                    lote_id += 1
                    lote_atual_id = lote_id
                    quantidade_acumulada = 0
            else:
                break  # Não há espaço disponível
    
    # Depois, processar lotes pequenos para tentar combiná-los em lotes existentes ou novos
    # Ordenar lotes pequenos do menor para o maior para maximizar o agrupamento
    for index, row in lotes_pequenos.sort_values('Qtd').iterrows():
        material_id = row['ID']
        material = row['Material']
        lote = row['Lote']
        quantidade = row['Qtd']
        qualidade = row['Qualidade']
        status = row['Status']
        localizacao = row['Localização']
        
        # Se o lote atual está cheio ou não iniciado, criar um novo
        if quantidade_acumulada >= quantidade_lote_max or quantidade_acumulada == 0:
            # Tentar combinar com lotes existentes que ainda têm espaço
            encontrou_espaco = False
            
            # Verificar lotes já criados para este grupo que ainda têm espaço
            lotes_existentes = [l for l in lotes_agrupados 
                                if l['ID'] == material_id and l['Quantidade'] < l['Max']]
            
            if lotes_existentes:
                # Ordenar por espaço disponível (do menor para o maior)
                # para otimizar o preenchimento de lotes
                lotes_existentes.sort(key=lambda x: x['Max'] - x['Quantidade'])
                
                for lote_existente in lotes_existentes:
                    espaco_disponivel = lote_existente['Max'] - lote_existente['Quantidade']
                    if quantidade <= espaco_disponivel:
                        # Cabe integralmente neste lote existente
                        lote_atual_id = lote_existente['Lote_id']
                        quantidade_acumulada = lote_existente['Quantidade']
                        encontrou_espaco = True
                        break
                    
            if not encontrou_espaco:
                # Se não encontrou espaço em lotes existentes, cria um novo
                lote_atual_id = lote_id
                quantidade_acumulada = 0
        
        quantidade_restante = quantidade
        while quantidade_restante > 0:
            espaco_disponivel = quantidade_lote_max - quantidade_acumulada
            quantidade_adicionar = min(espaco_disponivel, quantidade_restante)
            
            if quantidade_adicionar > 0:
                lotes_origem.append({
                    'ID': material_id,
                    'Material': material,
                    'Qualidade': qualidade,
                    'Quantidade': quantidade_adicionar,
                    'Lote_id': lote_atual_id,
                    'Localização': localizacao,
                    'Lote': lote,
                    'Status': status
                })
                
                quantidade_acumulada += quantidade_adicionar
                quantidade_restante -= quantidade_adicionar
                
                # Se o lote atual está completo, criar um novo lote
                if quantidade_acumulada >= quantidade_lote_max and quantidade_restante > 0:
                    lote_id += 1
                    lote_atual_id = lote_id
                    quantidade_acumulada = 0
            else:
                break
    
    # Criar novo lote para o próximo ID
    if quantidade_acumulada > 0:
        lote_id += 1
    
    # Registrar o lote agrupado se tiver quantidade > 0
    if quantidade_acumulada > 0:
        lotes_agrupados.append({
            'ID': material_id,
            'Material': material,
            'Qualidade': qualidade,
            'Quantidade': quantidade_acumulada,
            'Lote_id': lote_atual_id,
            'Max': quantidade_lote_max
        })

# Processar os dados de origem para criar os lotes agrupados consolidados
df_lotes_origem = pd.DataFrame(lotes_origem)

# Calcular os totais dos lotes agrupados a partir dos dados de origem
lotes_agrupados = []
for lote_id, grupo in df_lotes_origem.groupby('Lote_id'):
    if len(grupo) > 0:
        row = grupo.iloc[0]
        quantidade_max = estoque_filtrado.loc[estoque_filtrado['ID'] == row['ID'], 'Lote Max'].iloc[0]
        quantidade_total = grupo['Quantidade'].sum()
        
        # Verificação de segurança: garantir que nenhum lote ultrapasse o máximo
        if quantidade_total > quantidade_max:
            print(f"AVISO: Lote_id {lote_id} ultrapassa o máximo permitido: {quantidade_total} > {quantidade_max}")
            quantidade_total = quantidade_max
            
        lotes_agrupados.append({
            'ID': row['ID'],
            'Material': row['Material'],
            'Qualidade': row['Qualidade'],
            'Quantidade': quantidade_total,
            'Lote_id': lote_id,
            'Max': quantidade_max
        })

# Criar DataFrames finais
df_lotes_agrupados = pd.DataFrame(lotes_agrupados)

# Adicionar etapa final de verificação e correção para garantir que nenhum lote ultrapassa o máximo
for idx, row in df_lotes_agrupados.iterrows():
    if row['Quantidade'] > row['Max']:
        print(f"CORREÇÃO: Lote_id {row['Lote_id']} ajustado de {row['Quantidade']} para {row['Max']}")
        df_lotes_agrupados.at[idx, 'Quantidade'] = row['Max']
        
        # Ajustar também os lotes de origem, caso necessário
        qtd_excesso = row['Quantidade'] - row['Max']
        lotes_origem_afetados = df_lotes_origem[df_lotes_origem['Lote_id'] == row['Lote_id']]
        if not lotes_origem_afetados.empty:
            idx_ultimo = lotes_origem_afetados.index[-1]
            qtd_ultimo = df_lotes_origem.at[idx_ultimo, 'Quantidade']
            
            if qtd_ultimo > qtd_excesso:
                # Reduzir apenas a quantidade do último lote
                df_lotes_origem.at[idx_ultimo, 'Quantidade'] = qtd_ultimo - qtd_excesso
            else:
                # Necessário ajustar mais de um lote
                df_lotes_origem.at[idx_ultimo, 'Quantidade'] = 0
                qtd_restante = qtd_excesso - qtd_ultimo
                
                # Percorrer os lotes de trás para frente reduzindo as quantidades
                for i in reversed(lotes_origem_afetados.index[:-1]):
                    if qtd_restante <= 0:
                        break
                    
                    qtd_atual = df_lotes_origem.at[i, 'Quantidade']
                    if qtd_atual > qtd_restante:
                        df_lotes_origem.at[i, 'Quantidade'] = qtd_atual - qtd_restante
                        qtd_restante = 0
                    else:
                        df_lotes_origem.at[i, 'Quantidade'] = 0
                        qtd_restante -= qtd_atual

# Validar se nenhum lote ultrapassa o máximo permitido
df_lotes_agrupados['Excede_Max'] = df_lotes_agrupados['Quantidade'] > df_lotes_agrupados['Max']
if df_lotes_agrupados['Excede_Max'].any():
    print("ATENÇÃO: Ainda existem lotes que excedem a quantidade máxima após correções!")
    print(df_lotes_agrupados[df_lotes_agrupados['Excede_Max']])

# Remover coluna de validação antes de salvar
df_lotes_agrupados = df_lotes_agrupados.drop(columns=['Excede_Max'])

# Salvar os resultados
df_lotes_agrupados.to_excel('new_datasets/lotes_agrupados.xlsx', index=False)
df_lotes_origem.to_excel('new_datasets/lotes_agrupados_origem.xlsx', index=False)

# Exibir os primeiros resultados para verificação
print("Lotes Agrupados:")
print(df_lotes_agrupados.head())

print("\nLotes Originais:")
print(df_lotes_origem.head())