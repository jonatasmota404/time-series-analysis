import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from helpers import salvar_metricas_em_csv, salva_previsao_csv
import numpy as np

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
