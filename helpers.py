# helpers.py
import os
import pandas as pd

def salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, granularidade, pasta_metricas="metricas", arquivo_resultados="resultados_modelos.csv"):
    """
    Salva as métricas de um modelo específico em um arquivo CSV dentro de uma pasta dedicada.
    Substitui as métricas do modelo se ele já existir no arquivo.

    Args:
        nome_modelo (str): Nome do modelo.
        mae (float): Erro Médio Absoluto.
        rmse (float): Raiz do Erro Quadrático Médio.
        r2 (float): Coeficiente de Determinação.
        granularidade (str): Tipo de granularidade utilizada (diaria, semanal, mensal).
        pasta_metricas (str): Nome da pasta para salvar as métricas.
        arquivo_resultados (str): Nome do arquivo de métricas consolidado.
    """
    os.makedirs(pasta_metricas, exist_ok=True)  # Criar pasta se não existir
    caminho_arquivo = os.path.join(pasta_metricas, arquivo_resultados)

    df_resultado = pd.DataFrame({
        "Modelo": [f"{nome_modelo}_{granularidade}"],
        "Granularidade": [granularidade],
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2]
    })

    try:
        if os.path.exists(caminho_arquivo):
            df_existente = pd.read_csv(caminho_arquivo)
            df_existente = df_existente[df_existente["Modelo"] != f"{nome_modelo}_{granularidade}"]
            df_atualizado = pd.concat([df_existente, df_resultado], ignore_index=True)
        else:
            df_atualizado = df_resultado

        df_atualizado.to_csv(caminho_arquivo, index=False)
        print(f"✅ Resultados do {nome_modelo}_{granularidade} salvos em '{caminho_arquivo}'.")

    except Exception as e:
        print(f"❌ Erro ao salvar métricas: {e}")

def salva_previsao_csv(nome_modelo, predictions, granularidade, pasta_resultados="resultados", caminho_teste="dados_processados"):
    """
    Salva os dados de teste e as previsões em arquivos CSV dentro de uma pasta dedicada, garantindo consistência no formato das colunas.

    Args:
        nome_modelo (str): Nome do modelo.
        predictions (array-like): Previsões geradas pelo modelo.
        granularidade (str): Tipo de granularidade utilizada (diaria, semanal, mensal).
        pasta_resultados (str): Nome da pasta para salvar os arquivos de previsão.
        caminho_teste (str): Caminho para o diretório contendo os arquivos de teste.
    """
    os.makedirs(pasta_resultados, exist_ok=True)  # Criar pasta se não existir

    caminho_previsoes = os.path.join(pasta_resultados, f"{nome_modelo}_{granularidade}_predictions.csv")
    caminho_arquivo_teste = os.path.join(caminho_teste, f"test_data_{granularidade}.csv")

    try:
        # Carregar os dados de teste da granularidade correta
        if not os.path.exists(caminho_arquivo_teste):
            raise FileNotFoundError(f"Arquivo de teste '{caminho_arquivo_teste}' não encontrado.")

        test_data = pd.read_csv(caminho_arquivo_teste)

        # Validar se a quantidade de previsões bate com os dados de teste
        if len(predictions) != len(test_data):
            print(f"⚠️ Tamanho do conjunto de teste: {len(test_data)}, tamanho das previsões: {len(predictions)}")
            raise ValueError("O número de previsões não corresponde ao número de registros no conjunto de teste.")

        # Criar DataFrame com previsões
        df_previsoes = pd.DataFrame({
            "Data": test_data['Data'],
            "Preco_Real": test_data['Preco_Medio'],
            "Previsao": predictions
        })

        df_previsoes.to_csv(caminho_previsoes, index=False)
        print(f"✅ Previsões do modelo {nome_modelo}_{granularidade} salvas em '{caminho_previsoes}'.")

    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
    except ValueError as e:
        print(f"❌ Erro: {e}")

def tratar_nans(*dfs, metodo="interpolacao"):
    """
    Trata valores ausentes (NaN) em um ou mais DataFrames.

    Parâmetros:
    - dfs (tuple de DataFrames): Um ou mais DataFrames a serem tratados.
    - metodo (str): Método de preenchimento ('interpolacao', 'ffill' ou 'drop').

    Retorna:
    - Tupla de DataFrames tratados (se um único DataFrame for passado, retorna um único DataFrame).
    """
    dfs_tratados = []

    for i, df in enumerate(dfs):
        if df.isna().sum().any():
            print(f"⚠️ Valores ausentes encontrados no DataFrame {i + 1}. Aplicando método: {metodo}...")

            if metodo == "interpolacao":
                df.interpolate(method='linear', inplace=True)
                print(f"✅ Interpolação aplicada com sucesso no DataFrame {i + 1}.")

            elif metodo == "ffill":
                df.fillna(method='ffill', inplace=True)
                print(f"✅ Forward Fill aplicado com sucesso no DataFrame {i + 1}.")

            elif metodo == "drop":
                df.dropna(inplace=True)
                print(f"✅ Linhas com NaN removidas no DataFrame {i + 1}.")

        else:
            print(f"✅ Nenhum valor ausente encontrado no DataFrame {i + 1}.")

        dfs_tratados.append(df)

    # Se houver um único DataFrame, retorna ele diretamente; caso contrário, retorna uma tupla
    return dfs_tratados if len(dfs_tratados) > 1 else dfs_tratados[0]

