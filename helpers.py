import os
import pandas as pd

def salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, pasta_metricas="metricas", arquivo_resultados="resultados_modelos.csv"):
    """
    Salva as métricas de um modelo específico em um arquivo CSV dentro de uma pasta dedicada.

    Args:
        nome_modelo (str): Nome do modelo.
        mae (float): Erro Médio Absoluto.
        rmse (float): Raiz do Erro Quadrático Médio.
        r2 (float): Coeficiente de Determinação.
        pasta_metricas (str): Nome da pasta para salvar as métricas.
        arquivo_resultados (str): Nome do arquivo de métricas consolidado.
    """
    # Criar a pasta de métricas, se não existir
    os.makedirs(pasta_metricas, exist_ok=True)

    # Caminho completo para o arquivo de métricas
    caminho_arquivo = os.path.join(pasta_metricas, arquivo_resultados)

    # Criar um DataFrame com as métricas
    df_resultado = pd.DataFrame({
        "Modelo": [nome_modelo],
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2]
    })

    # Salvar o DataFrame no arquivo CSV, no modo 'append'
    df_resultado.to_csv(caminho_arquivo, mode='a', index=False, header=not os.path.exists(caminho_arquivo))
    print(f"Resultados do {nome_modelo} adicionados a '{caminho_arquivo}'")


def salva_teste_previsao_csv(nome_modelo, test_data, predictions, pasta_testes="testes"):
    """
    Salva os dados de teste e as previsões em arquivos CSV dentro de uma pasta dedicada.

    Args:
        nome_modelo (str): Nome do modelo.
        test_data (DataFrame): Dados de teste.
        predictions (array-like): Previsões geradas pelo modelo.
        pasta_testes (str): Nome da pasta para salvar os arquivos de teste e previsão.
    """
    # Criar a pasta de testes, se não existir
    os.makedirs(pasta_testes, exist_ok=True)

    # Caminhos completos para os arquivos de teste e previsões
    caminho_teste = os.path.join(pasta_testes, f"{nome_modelo}_test_data.csv")
    caminho_previsoes = os.path.join(pasta_testes, f"{nome_modelo}_predictions.csv")

    # Salvar o conjunto de teste e previsões
    test_data.to_csv(caminho_teste, index=False)
    pd.DataFrame({"Data": test_data['Data'], "Previsao": predictions}).to_csv(caminho_previsoes, index=False)

    print(f"Dados de teste e previsões do modelo {nome_modelo} salvos em '{pasta_testes}'.")
