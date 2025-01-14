# modelos/arima.py
from statsmodels.tsa.arima.model import ARIMA
from dados import carregar_dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_teste_previsao_csv

def executar_arima(caminho_arquivos, p=1, d=1, q=1):
    # Carregar dados de treino e teste
    train_data, test_data = carregar_dados(caminho_arquivos)
    
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
    salva_teste_previsao_csv("arima", test_data, predictions)
    
    # Salvar as métricas no CSV sem sobrescrever
    salvar_metricas_em_csv("ARIMA", mae, rmse, r2)
    
    return mae, rmse, r2

