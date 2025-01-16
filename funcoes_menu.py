import modelos as model
from visualizacao import plot_comparacao_modelo
import pandas as pd
import os

# Mapear índices aos nomes dos modelos
modelos_disponiveis = {
    "1": "Regressão Linear",
    "2": "Random Forest",
    "3": "ARIMA",
    "4": "SARIMA",
    "5": "Prophet"
}

arquivos_prediction = {
      "1": "regressao_linear",
      "2": "random_forest",
      "3": "arima",
      "4": "sarima",
      "5": "prophet"
    }

# Função para executar um modelo específico
def executar_modelo_especifico(indice_modelo):
    if indice_modelo == "1":
        model.executar_regressao_linear()
    elif indice_modelo == "2":
        model.executar_random_forest()
    elif indice_modelo == "3":
        model.executar_arima(p=1, d=1, q=1)
    elif indice_modelo == "4":
        model.executar_sarima(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
    elif indice_modelo == "5":
        model.executar_prophet()
    else:
        print("\nÍndice inválido. Escolha um modelo da lista.")

# Função para exibir métricas de um modelo específico
def exibir_metricas(indice_modelo, caminho_pasta="metricas"):
    nome_modelo = modelos_disponiveis.get(indice_modelo)
    if nome_modelo:
        try:
            # Ler o arquivo consolidado de métricas
            caminho_metricas = f"./{caminho_pasta}/resultados_modelos.csv"
            resultados = pd.read_csv(caminho_metricas)
            metricas = resultados[resultados["Modelo"] == nome_modelo]
            if not metricas.empty:
                print(f"\nMétricas para o modelo {nome_modelo}:\n")
                print(metricas.to_string(index=False))
            else:
                print(f"\nNenhum resultado encontrado para o modelo {nome_modelo}.")
        except FileNotFoundError:
            print(f"\nArquivo de métricas não encontrado. Execute os modelos primeiro.")
    else:
        print("\nÍndice inválido. Escolha um modelo da lista.")

# Função para exibir gráficos de um modelo específico
def exibir_graficos(indice_modelo):
    nome_modelo = modelos_disponiveis.get(indice_modelo)
    arquivo_predicao = arquivos_prediction.get(indice_modelo)
    if nome_modelo:
        try:
            # Construir caminhos para os arquivos de previsões
            predictions_file = f"./resultados/{arquivo_predicao}_predictions.csv"
            # Gerar o gráfico
            plot_comparacao_modelo(nome_modelo, predictions_file)
        except FileNotFoundError:
            print(f"\nArquivos para o modelo {nome_modelo} não encontrados. Execute o modelo primeiro.")
    else:
        print("\nÍndice inválido. Escolha um modelo da lista.")

# Menu interativo
def menu_interativo():
    while True:
        print("\n=== MENU INTERATIVO ===")
        print("1. Executar análise de um modelo específico")
        print("2. Mostrar gráficos de um modelo específico")
        print("3. Mostrar métricas de um modelo específico")
        print("4. Executar todos os modelos de uma só vez")
        print("5. Sair")

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            print("\nModelos disponíveis:")
            for indice, modelo in modelos_disponiveis.items():
                print(f"{indice}. {modelo}")
            indice_modelo = input("Digite o índice do modelo desejado: ")
            executar_modelo_especifico(indice_modelo)

        elif escolha == "2":
            print("\nModelos disponíveis:")
            for indice, modelo in modelos_disponiveis.items():
                print(f"{indice}. {modelo}")
            indice_modelo = input("Digite o índice do modelo desejado: ")
            exibir_graficos(indice_modelo)

        elif escolha == "3":
            print("\nModelos disponíveis:")
            for indice, modelo in modelos_disponiveis.items():
                print(f"{indice}. {modelo}")
            indice_modelo = input("Digite o índice do modelo desejado: ")
            exibir_metricas(indice_modelo)

        elif escolha == "4":
            print("\nExecutando todos os modelos...")
            model.executar_regressao_linear()
            model.executar_random_forest()
            model.executar_arima(p=1, d=1, q=1)
            model.executar_sarima(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
            model.executar_prophet()
            print("\nTodos os modelos foram executados com sucesso.")

        elif escolha == "5":
            print("Saindo do programa.")
            break

        else:
            print("Opção inválida. Tente novamente.")
