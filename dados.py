import pandas as pd
import glob
import numpy as np
import os

def carregar_dados(caminho_arquivos="../data/ca-*.csv", pasta="dados_processados"):
    """
    Carrega os dados, filtra gasolina e gera as granularidades mensal, semanal e diária,
    dividindo cada uma em conjuntos de treino e teste.

    Args:
        caminho_arquivos (str): Caminho dos arquivos CSV.
        pasta (str): Diretório onde os arquivos processados serão salvos.

    Returns:
        dict: Dicionário contendo os DataFrames de treino e teste para cada granularidade.
    """
    # Lista para armazenar cada DataFrame carregado
    dataframes = []
    for arquivo in glob.glob(caminho_arquivos):
        # Carregar o arquivo CSV
        df = pd.read_csv(arquivo, delimiter=';')
        # Filtrar somente gasolina e colunas de interesse
        df = df[df['Produto'] == 'GASOLINA'][['Data da Coleta', 'Valor de Venda']]
        df['Data da Coleta'] = pd.to_datetime(df['Data da Coleta'], dayfirst=True, errors='coerce')
        df['Valor de Venda'] = df['Valor de Venda'].str.replace(',', '.').astype(float)
        dataframes.append(df)

    # Concatenar todos os DataFrames
    dados_completos = pd.concat(dataframes, ignore_index=True)

    # Criar a pasta de destino, se não existir
    pasta_destino = f"./{pasta}"
    os.makedirs(pasta_destino, exist_ok=True)

    # Função auxiliar para calcular a média e dividir em treino e teste
    def processar_granularidade(df, frequencia, nome):
        # Agrupar por média na frequência especificada
        df_resample = df.resample(frequencia, on='Data da Coleta')['Valor de Venda'].mean().reset_index()
        df_resample.columns = ['Data', 'Preco_Medio']

        # Dividir em treino (80%) e teste (20%)
        train_size = int(len(df_resample) * 0.8)
        train_data = df_resample[:train_size].copy()
        test_data = df_resample[train_size:].copy()

        # Adicionar a coluna de índice temporal
        train_data['Time_Index'] = np.arange(len(train_data))
        test_data['Time_Index'] = np.arange(len(train_data), len(df_resample))

        # Salvar os arquivos na pasta especificada
        train_data.to_csv(os.path.join(pasta_destino, f"train_data_{nome}.csv"), index=False)
        test_data.to_csv(os.path.join(pasta_destino, f"test_data_{nome}.csv"), index=False)

        print(f"Granularidade {nome} processada e salva em {pasta_destino}")
        return train_data, test_data

    # Gerar granularidades e salvar
    granularidades = {
        "mensal": "ME",
        "semanal": "W",
        "diaria": "D"
    }

    resultados = {}
    for nome, freq in granularidades.items():
        resultados[nome] = processar_granularidade(dados_completos, freq, nome)

    return resultados

# Caminho dos arquivos CSV (ajuste para o caminho correto)
#caminho_arquivos = "../data/ca-*.csv"

# Executar o processamento
#resultados = carregar_dados(caminho_arquivos)
