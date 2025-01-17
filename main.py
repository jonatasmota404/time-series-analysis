from funcoes_menu import menu_interativo
from dados import carregar_dados
import os

caminho_arquivos = "../data/ca-*.csv"
pasta_dados_processados = "dados_processados"

# Execução principal
if __name__ == "__main__":
    # Verificar se a pasta de dados processados existe
    if not os.path.exists(f"./{pasta_dados_processados}"):
        print("Carregando os dados...\n\n")
        carregar_dados(caminho_arquivos, pasta_dados_processados)
    menu_interativo()
