# modelos/prophet.py
from prophet import Prophet
from dados import carregar_dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_teste_previsao_csv

def executar_prophet(caminho_arquivos):
    # Carregar dados de treino e teste
    train_data, test_data = carregar_dados(caminho_arquivos)
    
    # Preparar os dados no formato que o Prophet espera
    train_df = train_data.rename(columns={"Data": "ds", "Preco_Medio": "y"})
    test_df = test_data.rename(columns={"Data": "ds", "Preco_Medio": "y"})
    
    # Configurar e treinar o modelo Prophet
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_prophet.fit(train_df)
    
    # Fazer previsões para o período do conjunto de teste
    future = test_df[['ds']]
    forecast = model_prophet.predict(future)
    
    # Extrair as previsões
    predictions = forecast['yhat'].values
    
    # Calcular métricas de avaliação
    mae = mean_absolute_error(test_df['y'], predictions)
    rmse = np.sqrt(mean_squared_error(test_df['y'], predictions))
    r2 = r2_score(test_df['y'], predictions)
    
    # Salvar as métricas e os dados de teste e previsões usando as funções de helpers.py
    salvar_metricas_em_csv("Prophet", mae, rmse, r2)
    salva_teste_previsao_csv("prophet", test_data, predictions)
