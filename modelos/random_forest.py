# modelos/random_forest.py
from sklearn.ensemble import RandomForestRegressor
from dados import carregar_dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from helpers import salvar_metricas_em_csv, salva_teste_previsao_csv
import numpy as np

def executar_random_forest(caminho_arquivos):
    train_data, test_data = carregar_dados(caminho_arquivos)
    
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
    salva_teste_previsao_csv("random_forest", test_data, predictions)
    
    
    return mae, rmse, r2
