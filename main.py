from funcoes_menu import menu_interativo
from dados import carregar_dados
import os

caminho_arquivos = "../data/ca-*.csv"
pasta_dados_processados = "dados_processados"

def verificar_dados_processados(pasta):
    """
    Verifica se os arquivos de treino para todas as granularidades estão na pasta de dados processados.
    """
    arquivos_necessarios = [
        "train_data_mensal.csv",
        "train_data_semanal.csv",
        "train_data_diaria.csv"
    ]
    return all(os.path.exists(os.path.join(pasta, arquivo)) for arquivo in arquivos_necessarios)

# Execução principal
if __name__ == "__main__":
    # Se os dados ainda não estiverem processados, executa o carregamento
    if not verificar_dados_processados(pasta_dados_processados):
        print("🔄 Carregando os dados...\n")
        carregar_dados(caminho_arquivos, pasta_dados_processados)
        print("✅ Dados processados e salvos na pasta 'dados_processados'.\n")
    else:
        print("📂 Dados já processados encontrados.\n")

    # Iniciar o menu interativo
    menu_interativo()
