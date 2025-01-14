from modelos.regressao_linear import executar_regressao_linear
from modelos.random_forest import executar_random_forest
from modelos.arima import executar_arima
from modelos.sarima import executar_sarima
from modelos.prophet import executar_prophet
from visualizacao import plot_comparacao_modelo

def exibir_resultados(caminho_arquivos):
    # Regressão Linear
    #mae_lr, rmse_lr, r2_lr = executar_regressao_linear(caminho_arquivos)
    #print(f"Regressão Linear - MAE: {mae_lr}, RMSE: {rmse_lr}, R²: {r2_lr}")
    
    # Random Forest
    #mae_rf, rmse_rf, r2_rf = executar_random_forest(caminho_arquivos)
    #print(f"Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}, R²: {r2_rf}")
    
    # ARIMA
    #mae_arima, rmse_arima, r2_arima = executar_arima(caminho_arquivos, p=1, d=1, q=1)
    #print(f"ARIMA - MAE: {mae_arima}, RMSE: {rmse_arima}, R²: {r2_arima}")
    
    # SARIMA
    #mae_sarima, rmse_sarima, r2_sarima = executar_sarima(caminho_arquivos, p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
    #print(f"SARIMA - MAE: {mae_sarima}, RMSE: {rmse_sarima}, R²: {r2_sarima}")
    
    # Prophet
    mae_prophet, rmse_prophet, r2_prophet = executar_prophet(caminho_arquivos)
    print(f"Prophet - MAE: {mae_prophet}, RMSE: {rmse_prophet}, R²: {r2_prophet}")
