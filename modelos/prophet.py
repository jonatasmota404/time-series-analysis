import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_previsao_csv

def executar_prophet(caminho_arquivos="dados_processados"):
    """
    Executa o modelo Prophet utilizando os dados de treino e teste previamente salvos.
    Salva as métricas e previsões em arquivos específicos para posterior análise.
    """
    try:
        # Carregar os dados de treino e teste salvos
        train_data = pd.read_csv(f"./{caminho_arquivos}/train_data.csv")
        test_data = pd.read_csv(f"./{caminho_arquivos}/test_data.csv")
    except FileNotFoundError:
        print("Erro: Arquivos de treino e teste não encontrados. Certifique-se de preparar os dados antes de executar o modelo.")
        return

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

    # Salvar os resultados
    salvar_metricas_em_csv("Prophet", mae, rmse, r2)
    salva_previsao_csv("prophet", predictions)

    # Exibir as métricas no terminal
    print(f"Prophet - MAE: {mae}, RMSE: {rmse}, R²: {r2}")
