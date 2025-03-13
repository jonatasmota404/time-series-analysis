import modelos as model
import visualizacao as vlz
import pandas as pd
import os

# Mapear índices aos nomes dos modelos
modelos_disponiveis = {
    "1": "ARIMA",
    "2": "SARIMA",
    "3": "Prophet"
}

# Opções de granularidade
granularidades_disponiveis = {
    "1": "diaria",
    "2": "semanal",
    "3": "mensal"
}

# Função para selecionar a granularidade
def selecionar_granularidade():
    print("\n📊 Selecione a granularidade desejada:")
    for indice, gran in granularidades_disponiveis.items():
        print(f"{indice}. {gran.capitalize()}")
    escolha = input("Escolha a granularidade (1-3): ")
    return granularidades_disponiveis.get(escolha, "mensal")

# Função para executar um modelo específico com granularidade
def executar_modelo_especifico(indice_modelo):
    granularidade = selecionar_granularidade()  # Pergunta a granularidade ao usuário
    
    if indice_modelo == "1":
        model.executar_arima(granularidade=granularidade)
    elif indice_modelo == "2":
        model.executar_sarima(granularidade=granularidade)
    elif indice_modelo == "3":
        model.executar_prophet(granularidade=granularidade)
    else:
        print("\n❌ Índice inválido. Escolha um modelo da lista.")

# Menu interativo
def menu_interativo(nome_pasta_metricas="metricas", nome_pasta_test_data="dados_processados"):
    while True:
        print("\n=== 📊 MENU INTERATIVO 📊 ===")
        print("1. Executar análise de um modelo específico")
        print("2. Executar Análise Exploratória (EDA)")
        print("3. Exibir todas as métricas de um modelo específico")
        print("4. Comparar desempenho entre os modelos")
        print("Z. Sair")

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            print("\n📌 Modelos disponíveis:")
            for indice, modelo in modelos_disponiveis.items():
                print(f"{indice}. {modelo}")
            indice_modelo = input("Digite o índice do modelo desejado: ")
            executar_modelo_especifico(indice_modelo)

        elif escolha == "2":
            print("\n🔎 Executando Análise Exploratória (EDA)...")
            vlz.executar_eda()
            print("\n✅ EDA concluída com sucesso.")

        elif escolha == "3":
            print("\n📌 Modelos disponíveis:")
            for indice, modelo in modelos_disponiveis.items():
                print(f"{indice}. {modelo}")
            indice_modelo = input("Digite o índice do modelo desejado: ")
            nome_modelo = modelos_disponiveis.get(indice_modelo)

            if nome_modelo:
                print(f"\n📊 Exibindo todas as métricas para '{nome_modelo}'...")
                vlz.exibir_todas_as_metricas(nome_modelo)
            else:
                print("❌ Índice inválido. Tente novamente.")

        elif escolha == "4":
            print("\n🔍 Gerando análise comparativa entre os modelos...")
            vlz.comparar_modelos()
            print("\n✅ Comparação concluída e gráficos salvos.")

        elif escolha == "z" or escolha=="Z":
            print("👋 Saindo do programa.")
            break

        else:
            print("❌ Opção inválida. Tente novamente.")
