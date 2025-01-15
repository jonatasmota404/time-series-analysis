# modelos/regressao_linear.py
from sklearn.linear_model import LinearRegression
from dados import carregar_dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from helpers import salvar_metricas_em_csv, salva_teste_previsao_csv
import numpy as np

def executar_regressao_linear(caminho_arquivos):
    train_data, test_data = carregar_dados(caminho_arquivos)
    
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
    salva_teste_previsao_csv("regressao_linear", test_data, predictions)
    
