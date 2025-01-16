import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_previsao_csv

def executar_arima(caminho_arquivos="dados_processados", p=1, d=1, q=1):
    """
    Executa o modelo ARIMA nos dados de treino e teste previamente salvos.

    Args:
        caminho_arquivos (str, optional): Nome da pasta com os dados de treinamento e teste.
        p (int): Parâmetro auto-regressivo.
        d (int): Parâmetro de diferenciação.
        q (int): Parâmetro de média móvel.

    """
    # Ler os conjuntos de dados de treinamento e teste salvos
    try:
        train_data = pd.read_csv(f"./{caminho_arquivos}/train_data.csv")
        test_data = pd.read_csv(f"./{caminho_arquivos}/test_data.csv")
    except FileNotFoundError:
        print("Erro: Arquivos de treino e teste não encontrados. Certifique-se de executar 'carregar_dados' antes.")
        return

    # Configurar e treinar o modelo ARIMA com os parâmetros (p, d, q)
    model_arima = ARIMA(train_data['Preco_Medio'], order=(p, d, q))
    model_arima_fit = model_arima.fit()

    # Fazer previsões para o período do conjunto de teste
    predictions = model_arima_fit.forecast(steps=len(test_data))

    # Calcular métricas de avaliação
    mae = mean_absolute_error(test_data['Preco_Medio'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Preco_Medio'], predictions))
    r2 = r2_score(test_data['Preco_Medio'], predictions)

    # Salvar o conjunto de teste e previsões em CSV para análise posterior
    salva_previsao_csv("arima", predictions)

    # Salvar as métricas no CSV sem sobrescrever
    salvar_metricas_em_csv("ARIMA", mae, rmse, r2)

    # Exibir as métricas no terminal
    print(f"ARIMA - MAE: {mae}, RMSE: {rmse}, R²: {r2}")
