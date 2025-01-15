import pandas as pd
import glob
import numpy as np
import os

def carregar_dados(caminho_arquivos, pasta="dados_processados"):
    # Lista para armazenar cada DataFrame carregado
    dataframes = []
    for arquivo in glob.glob(caminho_arquivos):
        # Carregar o arquivo CSV
        df = pd.read_csv(arquivo, delimiter=';')
        # Filtrar somente gasolina e colunas de interesse (ajuste de acordo com o formato)
        df = df[df['Produto'] == 'GASOLINA'][['Data da Coleta', 'Valor de Venda']]
        df['Data da Coleta'] = pd.to_datetime(df['Data da Coleta'], dayfirst=True, errors='coerce')
        df['Valor de Venda'] = df['Valor de Venda'].str.replace(',', '.').astype(float)
        dataframes.append(df)

    # Agregar por média mensal
    dados_completos = pd.concat(dataframes, ignore_index=True)
    dados_completos['Ano_Mes'] = dados_completos['Data da Coleta'].dt.to_period('M')
    monthly_avg_prices = dados_completos.groupby('Ano_Mes')['Valor de Venda'].mean().reset_index()
    monthly_avg_prices['Ano_Mes'] = monthly_avg_prices['Ano_Mes'].dt.to_timestamp()
    monthly_avg_prices.columns = ['Data', 'Preco_Medio']

    
    # Dividir em treino e teste
    # Definir o tamanho do conjunto de treino (80% dos dados)
    train_size = int(len(monthly_avg_prices) * 0.8)
    # Dividir o DataFrame em conjuntos de treino e teste
    train_data = monthly_avg_prices[:train_size].copy()
    test_data = monthly_avg_prices[train_size:].copy()
    
    # Converter datas para índices numéricos
    train_data['Time_Index'] = np.arange(len(train_data))
    test_data['Time_Index'] = np.arange(len(train_data), len(monthly_avg_prices))

    # Definir o caminho da pasta para salvar os arquivos
    pasta_destino = f"./{pasta}"

    # Criar a pasta, se ela não existir
    os.makedirs(pasta_destino, exist_ok=True)

    # Salvar os arquivos na pasta especificada
    train_data.to_csv(os.path.join(pasta_destino, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(pasta_destino, "test_data.csv"), index=False)
    
    return train_data, test_data