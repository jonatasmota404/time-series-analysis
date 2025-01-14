# main.py
from resultados import exibir_resultados
from visualizacao import plot_comparacao_modelo

if __name__ == "__main__":
    #caminho_arquivos = "../data/ca-*.csv"
    #exibir_resultados(caminho_arquivos)

    # Gerar gráficos para cada modelo
    #plot_comparacao_modelo("Regressão Linear", "regressao_linear_test_data.csv", "regressao_linear_predictions.csv")
    #plot_comparacao_modelo("Random Forest", "random_forest_test_data.csv", "random_forest_predictions.csv")
    #plot_comparacao_modelo("ARIMA", "arima_test_data.csv", "arima_predictions.csv")
    #plot_comparacao_modelo("SARIMA", "sarima_test_data.csv", "sarima_predictions.csv")
    plot_comparacao_modelo("Prophet", "prophet_test_data.csv", "prophet_predictions.csv")
