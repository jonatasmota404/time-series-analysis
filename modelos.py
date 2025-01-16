import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from helpers import salvar_metricas_em_csv, salva_previsao_csv
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

def executar_random_forest(caminho_arquivos="dados_processados"):
    """
    Executa o modelo Random Forest utilizando os dados de treino e teste previamente salvos.
    Salva as métricas e previsões em arquivos específicos para posterior análise.
    """
    try:
        # Carregar os dados de treino e teste salvos
        train_data = pd.read_csv(f"./{caminho_arquivos}/train_data.csv")
        test_data = pd.read_csv(f"./{caminho_arquivos}/test_data.csv")
    except FileNotFoundError:
        print("Erro: Arquivos de treino e teste não encontrados. Certifique-se de preparar os dados antes de executar o modelo.")
        return

    # Configuração e treino do modelo
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
    model_rf.fit(train_data[['Time_Index']], train_data['Preco_Medio'])

    # Fazer previsões e calcular métricas
    predictions = model_rf.predict(test_data[['Time_Index']])
    mae = mean_absolute_error(test_data['Preco_Medio'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Preco_Medio'], predictions))
    r2 = r2_score(test_data['Preco_Medio'], predictions)

    # Salvar as métricas no CSV sem sobrescrever
    salvar_metricas_em_csv("Random Forest", mae, rmse, r2)

    # Salvar o conjunto de teste e previsões em CSV para análise posterior
    salva_previsao_csv("random_forest", predictions)

    # Exibir as métricas no terminal
    print(f"Random Forest - MAE: {mae}, RMSE: {rmse}, R²: {r2}")

def executar_regressao_linear(caminho_arquivos="dados_processados"):
    """
    Executa o modelo de Regressão Linear utilizando os dados de treino e teste previamente salvos.
    Salva as métricas e previsões em arquivos específicos para posterior análise.
    """
    try:
        # Carregar os dados de treino e teste salvos
        train_data = pd.read_csv(f"./{caminho_arquivos}/train_data.csv")
        test_data = pd.read_csv(f"./{caminho_arquivos}/test_data.csv")
    except FileNotFoundError:
        print("Erro: Arquivos de treino e teste não encontrados. Certifique-se de preparar os dados antes de executar o modelo.")
        return

    # Configuração e treino do modelo
    model_lr = LinearRegression()
    model_lr.fit(train_data[['Time_Index']], train_data['Preco_Medio'])

    # Fazer previsões e calcular métricas
    predictions = model_lr.predict(test_data[['Time_Index']])
    mae = mean_absolute_error(test_data['Preco_Medio'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Preco_Medio'], predictions))
    r2 = r2_score(test_data['Preco_Medio'], predictions)

    # Salvar as métricas no CSV sem sobrescrever
    salvar_metricas_em_csv("Regressão Linear", mae, rmse, r2)

    # Salvar o conjunto de teste e previsões em CSV para análise posterior
    salva_previsao_csv("regressao_linear", predictions)

    # Exibir as métricas no terminal
    print(f"Regressão Linear - MAE: {mae}, RMSE: {rmse}, R²: {r2}")

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
