import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_previsao_csv

def executar_sarima(caminho_arquivos="dados_processados", p=1, d=1, q=1, P=1, D=1, Q=1, s=12):
    """
    Executa o modelo SARIMA utilizando os dados de treino e teste previamente salvos.
    Salva as métricas e previsões em arquivos específicos para posterior análise.

    Args:
        caminho_arquivos (str): Caminho para a pasta contendo os dados salvos.
        p (int): Parâmetro auto-regressivo.
        d (int): Parâmetro de diferenciação.
        q (int): Parâmetro de média móvel.
        P (int): Componente sazonal auto-regressivo.
        D (int): Componente sazonal de diferenciação.
        Q (int): Componente sazonal de média móvel.
        s (int): Período sazonal.
    """
    try:
        # Carregar os dados de treino e teste salvos
        train_data = pd.read_csv(f"./{caminho_arquivos}/train_data.csv")
        test_data = pd.read_csv(f"./{caminho_arquivos}/test_data.csv")
    except FileNotFoundError:
        print("Erro: Arquivos de treino e teste não encontrados. Certifique-se de preparar os dados antes de executar o modelo.")
        return

    # Configurar e treinar o modelo SARIMA
    model_sarima = SARIMAX(
        train_data['Preco_Medio'],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_sarima_fit = model_sarima.fit(disp=False)

    # Fazer previsões para o período do conjunto de teste
    predictions = model_sarima_fit.forecast(steps=len(test_data))

    # Calcular métricas de avaliação
    mae = mean_absolute_error(test_data['Preco_Medio'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Preco_Medio'], predictions))
    r2 = r2_score(test_data['Preco_Medio'], predictions)

    # Salvar os resultados
    salvar_metricas_em_csv("SARIMA", mae, rmse, r2)
    salva_previsao_csv("sarima", predictions)

    # Exibir as métricas no terminal
    print(f"SARIMA - MAE: {mae}, RMSE: {rmse}, R²: {r2}")
