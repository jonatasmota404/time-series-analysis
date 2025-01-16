import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from helpers import salvar_metricas_em_csv, salva_previsao_csv
import numpy as np

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
