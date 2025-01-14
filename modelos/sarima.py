# modelos/arima.py
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dados import carregar_dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_teste_previsao_csv

def executar_sarima(caminho_arquivos, p=1, d=1, q=1, P=1, D=1, Q=1, s=12):
    # Carregar dados de treino e teste
    train_data, test_data = carregar_dados(caminho_arquivos)
    
    # Configurar e treinar o modelo SARIMA com os parâmetros (p, d, q) e (P, D, Q, s)
    model_sarima = SARIMAX(train_data['Preco_Medio'], order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
    model_sarima_fit = model_sarima.fit(disp=False)
    
    # Fazer previsões para o período do conjunto de teste
    predictions = model_sarima_fit.forecast(steps=len(test_data))
    
    # Calcular métricas de avaliação
    mae = mean_absolute_error(test_data['Preco_Medio'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Preco_Medio'], predictions))
    r2 = r2_score(test_data['Preco_Medio'], predictions)
    
    # Salvar o conjunto de teste e previsões em CSV para análise posterior
    salva_teste_previsao_csv("sarima", test_data, predictions)
    
    # Salvar as métricas no CSV sem sobrescrever
    salvar_metricas_em_csv("SARIMA", mae, rmse, r2)
    
    return mae, rmse, r2
