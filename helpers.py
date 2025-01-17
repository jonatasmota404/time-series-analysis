import os
import pandas as pd

def salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, pasta_metricas="metricas", arquivo_resultados="resultados_modelos.csv"):
    """
    Salva as métricas de um modelo específico em um arquivo CSV dentro de uma pasta dedicada.
    Substitui as métricas do modelo se ele já existir no arquivo.

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

    try:
        # Verificar se o arquivo já existe
        if os.path.exists(caminho_arquivo):
            df_existente = pd.read_csv(caminho_arquivo)

            # Substituir as métricas do modelo, se ele já existir
            df_existente = df_existente[df_existente["Modelo"] != nome_modelo]
            df_atualizado = pd.concat([df_existente, df_resultado], ignore_index=True)
        else:
            df_atualizado = df_resultado

        # Salvar o DataFrame atualizado no arquivo
        df_atualizado.to_csv(caminho_arquivo, index=False)
        print(f"Resultados do {nome_modelo} salvos em '{caminho_arquivo}'.")

    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")



def salva_previsao_csv(nome_modelo, predictions, pasta_resultados="resultados", caminho_teste="dados_processados"):
    """
    Salva os dados de teste e as previsões em arquivos CSV dentro de uma pasta dedicada,
    garantindo consistência no formato das colunas.

    Args:
        nome_modelo (str): Nome do modelo.
        predictions (array-like): Previsões geradas pelo modelo.
        pasta_resultados (str): Nome da pasta para salvar os arquivos de previsão.
        caminho_teste (str): Caminho para o diretório contendo o arquivo de teste.
    """
    # Criar a pasta de resultados, se não existir
    os.makedirs(pasta_resultados, exist_ok=True)

    # Caminho do arquivo de previsões
    caminho_previsoes = os.path.join(pasta_resultados, f"{nome_modelo}_predictions.csv")

    try:
        # Carregar os dados de teste
        test_data = pd.read_csv(f"./{caminho_teste}/test_data.csv")
        
        # Garantir que o número de previsões corresponda ao número de registros no conjunto de teste
        if len(predictions) != len(test_data):
            raise ValueError("O número de previsões não corresponde ao número de registros no conjunto de teste.")

        # Criar um DataFrame com as datas e previsões
        df_previsoes = pd.DataFrame({
            "Data": test_data['Data'],
            "Preco_Real": test_data['Preco_Medio'],
            "Previsao": predictions
        })

        # Salvar as previsões no arquivo CSV com as colunas padronizadas
        df_previsoes.to_csv(caminho_previsoes, index=False)
        print(f"Previsões do modelo {nome_modelo} salvas em '{caminho_previsoes}'.")

    except FileNotFoundError:
        print(f"Erro: Arquivo de teste não encontrado em '{caminho_teste}'. Certifique-se de que os dados estão disponíveis.")
    except ValueError as e:
        print(f"Erro: {e}")


