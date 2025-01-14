import pandas as pd
import matplotlib.pyplot as plt

def plot_comparacao_modelo(nome_modelo, arquivo_test_data, arquivo_predictions):
    # Carregar os dados de teste e previsões salvos em CSV
    test_data = pd.read_csv(arquivo_test_data)
    predictions = pd.read_csv(arquivo_predictions)
    
    # Plotar os valores reais vs. previsões
    plt.figure(figsize=(12, 6))
    plt.plot(test_data['Data'], test_data['Preco_Medio'], label="Valor Real", color='blue')
    plt.plot(test_data['Data'], predictions['Previsao'], label=f"Previsão - {nome_modelo}", color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel("Preço da Gasolina")
    plt.title(f"Comparação entre Valor Real e Previsão - {nome_modelo}")
    plt.legend()
    plt.show()