import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from helpers import tratar_nans
import os

# Garantir que as pastas existam
def garantir_pasta(caminho):
    os.makedirs(caminho, exist_ok=True)

def carregar_dados_granularidade(pasta="dados_processados"):
    """
    Carrega os dados processados nas três granularidades (mensal, semanal e diária).
    """
    granularidades = ["mensal", "semanal", "diaria"]
    dados = {}

    for nome in granularidades:
        caminho = os.path.join(pasta, f"train_data_{nome}.csv")
        if os.path.exists(caminho):
            dados[nome] = pd.read_csv(caminho)
            dados[nome]['Data'] = pd.to_datetime(dados[nome]['Data'])
        else:
            print(f"❌ Arquivo não encontrado: {caminho}")

    return dados

def estatisticas_descritivas(df, nome_granularidade):
    """
    Exibe estatísticas descritivas básicas do DataFrame.
    """
    print(f"\n📊 Estatísticas Descritivas - {nome_granularidade.capitalize()}")
    print(df.describe())

    # Verificar valores ausentes
    print(f"\n🔍 Valores ausentes: {df.isnull().sum().sum()} em {nome_granularidade}")

    # Verificar duplicatas
    print(f"📌 Duplicatas: {df.duplicated().sum()} em {nome_granularidade}")

def plotar_serie_temporal(df, nome_granularidade, pasta_saida="resultados_eda"):
    """
    Plota a série temporal com base na granularidade (mensal, semanal, diária) e salva o gráfico.
    """
    garantir_pasta(pasta_saida)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Data'], df['Preco_Medio'], label=f'Série {nome_granularidade.capitalize()}')
    plt.title(f'Série Temporal - {nome_granularidade.capitalize()}')
    plt.xlabel('Data')
    plt.ylabel('Preço Médio da Gasolina')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    caminho_saida = os.path.join(pasta_saida, f"serie_temporal_{nome_granularidade}.png")
    plt.savefig(caminho_saida)
    plt.close()

    print(f"📊 Gráfico da série temporal salvo em: {caminho_saida}")

def decompor_serie_temporal(df, nome_granularidade, modelo='additive', metodo_nan="interpolacao", pasta_saida="resultados_eda"):
    """
    Decompõe a série temporal em tendência, sazonalidade e resíduos e salva o gráfico.

    Parâmetros:
    - df (DataFrame): Dados a serem decompostos.
    - nome_granularidade (str): Nome da granularidade (ex.: "mensal", "semanal").
    - modelo (str): Tipo de modelo para decomposição ('additive' ou 'multiplicative').
    - metodo_nan (str): Método para lidar com NaNs ('interpolacao', 'ffill', 'drop').
    - pasta_saida (str): Caminho para salvar os gráficos gerados.
    """
    garantir_pasta(pasta_saida)

    df = df.set_index('Data')

    # Tratar valores ausentes com o método escolhido
    df = tratar_nans(df, metodo=metodo_nan)

    try:
        decomposicao = seasonal_decompose(df['Preco_Medio'], model=modelo, period=12)
        fig = decomposicao.plot()
        fig.suptitle(f"Decomposição da Série - {nome_granularidade.capitalize()}", fontsize=14)

        caminho_saida = os.path.join(pasta_saida, f"decomposicao_{nome_granularidade}.png")
        fig.savefig(caminho_saida)
        plt.close(fig)

        print(f"📊 Gráfico de decomposição salvo em: {caminho_saida}")

    except Exception as e:
        print(f"❌ Erro ao decompor a série em {nome_granularidade}: {e}")


    except Exception as e:
        print(f"❌ Erro ao decompor a série em {nome_granularidade}: {e}")

def plotar_histograma(df, nome_granularidade, pasta_saida="resultados_eda"):
    """
    Plota um histograma para visualizar a distribuição dos preços médios e salva o gráfico.
    """
    garantir_pasta(pasta_saida)

    plt.figure(figsize=(8, 4))
    sns.histplot(df['Preco_Medio'], kde=True, bins=30, color='royalblue')
    plt.title(f'Distribuição dos Preços - {nome_granularidade.capitalize()}')
    plt.xlabel('Preço Médio')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()

    caminho_saida = os.path.join(pasta_saida, f"histograma_{nome_granularidade}.png")
    plt.savefig(caminho_saida)
    plt.close()

    print(f"📊 Gráfico do histograma salvo em: {caminho_saida}")

def plotar_acf_pacf(df, nome_granularidade, lags=40, pasta_saida="resultados_eda"):
    """
    Plota os gráficos de Autocorrelação (ACF) e Autocorrelação Parcial (PACF) e salva em um único gráfico.
    """
    garantir_pasta(pasta_saida)

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    sm.graphics.tsa.plot_acf(df['Preco_Medio'], lags=lags, ax=plt.gca())
    plt.title(f"ACF - {nome_granularidade.capitalize()}")

    plt.subplot(122)
    sm.graphics.tsa.plot_pacf(df['Preco_Medio'], lags=lags, ax=plt.gca())
    plt.title(f"PACF - {nome_granularidade.capitalize()}")

    plt.tight_layout()

    caminho_saida = os.path.join(pasta_saida, f"acf_pacf_{nome_granularidade}.png")
    plt.savefig(caminho_saida)
    plt.close()

    print(f"📊 Gráficos de ACF e PACF salvos em: {caminho_saida}")

def executar_eda(pasta="dados_processados", pasta_saida="resultados_eda"):
    """
    Executa a Análise Exploratória de Dados (EDA) para cada granularidade.
    """
    dados = carregar_dados_granularidade(pasta)

    for nome_granularidade, df in dados.items():
        estatisticas_descritivas(df, nome_granularidade)
        plotar_serie_temporal(df, nome_granularidade, pasta_saida)
        decompor_serie_temporal(df, nome_granularidade, pasta_saida=pasta_saida)
        plotar_histograma(df, nome_granularidade, pasta_saida)
        plotar_acf_pacf(df, nome_granularidade, pasta_saida=pasta_saida)

    print("\n✅ Análise Exploratória concluída e gráficos salvos em:", pasta_saida)

def comparar_modelos(caminho_metricas="metricas/resultados_modelos.csv", pasta_resultados="resultados_comparacao"):
    """
    Compara os modelos ARIMA, SARIMA e Prophet em diferentes granularidades (diária, semanal, mensal).
    Gera gráficos comparativos para MAE, RMSE e R².
    
    Args:
        caminho_metricas (str): Caminho do arquivo CSV com as métricas dos modelos.
        pasta_resultados (str): Pasta para salvar os gráficos gerados.
    """
    # Garantir que a pasta de resultados exista
    os.makedirs(pasta_resultados, exist_ok=True)

    # Carregar as métricas
    try:
        df_metricas = pd.read_csv(caminho_metricas)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de métricas não encontrado em '{caminho_metricas}'.")
        return

    # Garantir que temos as colunas esperadas
    if not {"Modelo", "MAE", "RMSE", "R²"}.issubset(df_metricas.columns):
        print("❌ Erro: O arquivo de métricas está com formato incorreto.")
        return

    print("\n📊 Comparando o desempenho dos modelos...")

    # Separar granularidades
    granularidades = ["diaria", "semanal", "mensal"]

    # Iterar por cada métrica e gerar gráficos comparativos
    for metrica in ["MAE", "RMSE", "R²"]:
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        for granularidade in granularidades:
            # Filtrar dados da granularidade atual
            df_granularidade = df_metricas[df_metricas["Modelo"].str.contains(granularidade, case=False)]

            if df_granularidade.empty:
                print(f"⚠️ Nenhum dado encontrado para a granularidade '{granularidade}'.")
                continue

            # Gráfico de comparação para cada métrica
            sns.barplot(
                x="Modelo",
                y=metrica,
                data=df_granularidade,
                label=f"{granularidade.capitalize()}",
                dodge=True
            )

        plt.title(f"Comparação de {metrica} entre os Modelos por Granularidade")
        plt.xlabel("Modelos")
        plt.ylabel(metrica)
        plt.legend(title="Granularidade")
        
        caminho_grafico = os.path.join(pasta_resultados, f"comparacao_{metrica.lower()}.png")
        plt.savefig(caminho_grafico)
        plt.close()

        print(f"📊 Gráfico de {metrica} salvo em '{caminho_grafico}'.")

    print("\n✅ Comparação concluída com sucesso!")

def exibir_todas_as_metricas(nome_modelo, caminho_metricas="metricas/resultados_modelos.csv"):
    """
    Exibe todas as métricas (MAE, RMSE, R²) para um modelo específico em todas as granularidades.

    Args:
        nome_modelo (str): Nome do modelo a ser analisado (ex: "ARIMA", "SARIMA", "Prophet").
        caminho_metricas (str): Caminho do arquivo CSV com as métricas.
    """
    try:
        df_metricas = pd.read_csv(caminho_metricas)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de métricas não encontrado em '{caminho_metricas}'.")
        return

    # Filtrar métricas do modelo específico (independente de maiúsculas/minúsculas)
    df_modelo = df_metricas[df_metricas["Modelo"].str.contains(nome_modelo, case=False)]

    if df_modelo.empty:
        print(f"⚠️ Nenhum resultado encontrado para o modelo '{nome_modelo}'.")
        return

    print(f"\n📊 Métricas para o modelo '{nome_modelo}':\n")
    print(df_modelo.to_string(index=False))
