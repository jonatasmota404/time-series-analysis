import modelos as model
from visualizacao import plot_comparacao_modelo
import pandas as pd

# Mapear índices aos nomes dos modelos
modelos_disponiveis = {
    "1": "Regressão Linear",
    "2": "Random Forest",
    "3": "ARIMA",
    "4": "SARIMA",
    "5": "Prophet"
}

# Função para executar um modelo específico
def executar_modelo_especifico(indice_modelo, caminho_arquivos):
    nome_modelo = modelos_disponiveis.get(indice_modelo)
    if nome_modelo == "Regressão Linear":
        model.executar_regressao_linear(caminho_arquivos)
    elif nome_modelo == "Random Forest":
        model.executar_random_forest(caminho_arquivos)
    elif nome_modelo == "ARIMA":
        model.executar_arima(caminho_arquivos, p=1, d=1, q=1)
    elif nome_modelo == "SARIMA":
        model.executar_sarima(caminho_arquivos, p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
    elif nome_modelo == "Prophet":
        model.executar_prophet(caminho_arquivos)
    else:
        print("\nÍndice inválido. Escolha um modelo da lista.")

# Função para exibir métricas de um modelo específico
def exibir_metricas(indice_modelo):
    nome_modelo = modelos_disponiveis.get(indice_modelo)
    if nome_modelo:
        try:
            resultados = pd.read_csv("resultados_modelos.csv")
            metricas = resultados[resultados["Modelo"] == nome_modelo]
            if not metricas.empty:
                print(f"\nMétricas para o modelo {nome_modelo}:\n")
                print(metricas.to_string(index=False))
            else:
                print(f"\nNenhum resultado encontrado para o modelo {nome_modelo}.")
        except FileNotFoundError:
            print("\nArquivo 'resultados_modelos.csv' não encontrado. Execute os modelos primeiro.")
    else:
        print("\nÍndice inválido. Escolha um modelo da lista.")

# Função para exibir gráficos de um modelo específico
def exibir_graficos(indice_modelo):
    nome_modelo = modelos_disponiveis.get(indice_modelo)
    if nome_modelo:
        try:
            test_file = f"{nome_modelo.lower().replace(' ', '_')}_test_data.csv"
            predictions_file = f"{nome_modelo.lower().replace(' ', '_')}_predictions.csv"
            plot_comparacao_modelo(nome_modelo, test_file, predictions_file)
        except FileNotFoundError:
            print(f"\nArquivos para o modelo {nome_modelo} não encontrados. Execute o modelo primeiro.")
    else:
        print("\nÍndice inválido. Escolha um modelo da lista.")

# Menu interativo
def menu_interativo(caminho_arquivos):
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
            executar_modelo_especifico(indice_modelo, caminho_arquivos)

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
            model.executar_regressao_linear(caminho_arquivos)
            model.executar_random_forest(caminho_arquivos)
            model.executar_arima(caminho_arquivos, p=1, d=1, q=1)
            model.executar_sarima(caminho_arquivos, p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
            model.executar_prophet(caminho_arquivos)
            print("\nTodos os modelos foram executados com sucesso.")

        elif escolha == "5":
            print("Saindo do programa.")
            break

        else:
            print("Opção inválida. Tente novamente.")

# Execução principal
if __name__ == "__main__":
    caminho_arquivos = "../data/ca-*.csv"
    menu_interativo(caminho_arquivos)
