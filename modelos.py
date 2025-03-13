import os
import pandas as pd
import numpy as np
import helpers as helper
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def testar_estacionariedade(series):
    """
    Executa os testes de ADF e KPSS para avaliar a estacionariedade da série.
    """
    print("\n📊 Teste de Estacionariedade:")

    # Remover valores ausentes e infinitos
    series.dropna(inplace=True)
    series.replace([np.inf, -np.inf], np.nan, inplace=True)
    series.dropna(inplace=True)

    if series.empty:
        print("❌ Série vazia após limpeza. Verifique os dados.")
        return True

    # Teste de Dickey-Fuller (ADF)
    adf_resultado = adfuller(series)
    print(f"ADF Statistic: {adf_resultado[0]:.4f}, p-valor: {adf_resultado[1]:.4f}")
    adf_estacionaria = adf_resultado[1] < 0.05

    # Teste KPSS
    try:
        kpss_resultado, kpss_p, *_ = kpss(series, nlags="auto")
        print(f"KPSS Statistic: {kpss_resultado:.4f}, p-valor: {kpss_p:.4f}")
        kpss_estacionaria = kpss_p > 0.05
    except ValueError:
        print("⚠️ Erro ao executar o teste KPSS.")
        kpss_estacionaria = False

    return adf_estacionaria and kpss_estacionaria

def carregar_dados(caminho_teste="dados_processados", granularidade="mensal"):
    """
    Carrega os dados de treino e teste e verifica a existência dos arquivos.
    """
    try:
        train_data = pd.read_csv(f"./{caminho_teste}/train_data_{granularidade}.csv")
        test_data = pd.read_csv(f"./{caminho_teste}/test_data_{granularidade}.csv")
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos não encontrados para a granularidade {granularidade}.")
        return None, None

    # Converter 'Data' para datetime e definir como índice
    train_data["Data"] = pd.to_datetime(train_data["Data"])
    train_data.set_index("Data", inplace=True)

    test_data["Data"] = pd.to_datetime(test_data["Data"])
    test_data.set_index("Data", inplace=True)

    # Garantir frequência do índice para previsões futuras
    if granularidade == "mensal":
        train_data = train_data.asfreq("ME")  # Atualizado
        test_data = test_data.asfreq("ME")
    elif granularidade == "semanal":
        train_data = train_data.asfreq("W")
        test_data = test_data.asfreq("W")
    elif granularidade == "diaria":
        train_data = train_data.asfreq("D")
        test_data = test_data.asfreq("D")

    return train_data, test_data

def ajustar_arima(train_data, test_data, nome_modelo="ARIMA", granularidade="mensal"):
    """
    Ajusta o modelo ARIMA com detecção automática de diferenciação (d).
    """
    if train_data is None or test_data is None:
        print("❌ Erro: Conjuntos de dados inválidos.")
        return

    # Verificar se a série é estacionária
    precisa_diff = not testar_estacionariedade(train_data["Preco_Medio"])

    if precisa_diff:
        print("🔁 Aplicando diferenciação para tornar a série estacionária...")
        train_data["Preco_Medio"] = train_data["Preco_Medio"].diff()
        train_data.dropna(inplace=True)

    melhor_p, melhor_d, melhor_q = 1, 1, 1
    melhor_rmse = float("inf")

    print("\n🔍 Otimizando parâmetros para ARIMA...")
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    modelo = ARIMA(train_data["Preco_Medio"], order=(p, d, q))
                    resultado = modelo.fit()
                    previsoes = resultado.forecast(steps=len(test_data))

                    # Garantir o alinhamento do índice
                    previsoes.index = test_data.index

                    # Remover NaNs antes de calcular métricas
                    previsoes.dropna(inplace=True)
                    test_data_validado = test_data.loc[previsoes.index]

                    rmse = np.sqrt(mean_squared_error(test_data_validado["Preco_Medio"], previsoes))

                    if rmse < melhor_rmse:
                        melhor_p, melhor_d, melhor_q = p, d, q
                        melhor_rmse = rmse

                except Exception as e:
                    print(f"❌ Erro ao ajustar ARIMA({p},{d},{q}): {e}")

    print(f"✅ Melhor configuração para ARIMA: (p, d, q) = ({melhor_p}, {melhor_d}, {melhor_q})")

    # Ajustar o modelo final
    modelo_final = ARIMA(train_data["Preco_Medio"], order=(melhor_p, melhor_d, melhor_q))
    resultado_final = modelo_final.fit()

    # Fazer previsões e alinhar índice
    previsoes = resultado_final.forecast(steps=len(test_data))
    previsoes.index = test_data.index

    # Remover valores ausentes
    previsoes.dropna(inplace=True)
    test_data_validado = test_data.loc[previsoes.index]

    # Calcular métricas
    mae = mean_absolute_error(test_data_validado["Preco_Medio"], previsoes)
    rmse = np.sqrt(mean_squared_error(test_data_validado["Preco_Medio"], previsoes))
    r2 = r2_score(test_data_validado["Preco_Medio"], previsoes)

    # Salvar métricas e previsões
    helper.salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, granularidade)
    helper.salva_previsao_csv(nome_modelo.lower(), previsoes.values, granularidade)

    print(f"📊 {nome_modelo} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"📁 Previsões salvas em 'resultados/{nome_modelo.lower()}_predictions.csv'.")

def executar_arima(caminho_teste="dados_processados", granularidade="mensal"):
    """
    Executa o pipeline completo do modelo ARIMA para a granularidade especificada.
    """
    print(f"\n🚀 Executando ARIMA com otimização de parâmetros para granularidade {granularidade}...")

    train_data, test_data = carregar_dados(caminho_teste, granularidade)
    # Aplicar tratamento (escolha o método desejado: 'interpolacao', 'ffill' ou 'drop')
    train_data, test_data = helper.tratar_nans(train_data, test_data, metodo="interpolacao")
    ajustar_arima(train_data, test_data, "ARIMA", granularidade)

def ajustar_sarima(train_data, test_data, nome_modelo="SARIMA", granularidade="mensal"):
    """
    Ajusta o modelo SARIMA com detecção automática de diferenciação (d) e sazonalidade (s).
    """
    if train_data is None or test_data is None:
        print("❌ Erro: Conjuntos de dados inválidos.")
        return

    # Validar se há valores ausentes
    train_data, test_data = helper.tratar_nans(train_data, test_data, metodo="interpolacao")

    # Detectar a sazonalidade com base na granularidade
    sazonalidade = {"diaria": 7, "semanal": 52, "mensal": 12}.get(granularidade, 12)

    # Verificar se a série é estacionária
    precisa_diff = not testar_estacionariedade(train_data["Preco_Medio"])

    if precisa_diff:
        print("🔁 Aplicando diferenciação para tornar a série estacionária...")
        train_data["Preco_Medio"] = train_data["Preco_Medio"].diff().dropna()

    melhor_p, melhor_d, melhor_q = 1, 1, 1
    melhor_P, melhor_D, melhor_Q = 0, 0, 0
    melhor_rmse = float("inf")

    print("\n🔍 Otimizando parâmetros para SARIMA...")
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                for P in range(0, 2):
                    for D in range(0, 2):
                        for Q in range(0, 2):
                            try:
                                modelo = SARIMAX(
                                    train_data["Preco_Medio"],
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, sazonalidade),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                resultado = modelo.fit(disp=False)
                                previsoes = resultado.forecast(steps=len(test_data))

                                # Alinhar índice
                                previsoes.index = test_data.index

                                # Remover NaNs para calcular métricas corretamente
                                previsoes.dropna(inplace=True)
                                test_data_validado = test_data.loc[previsoes.index]

                                rmse = np.sqrt(mean_squared_error(test_data_validado["Preco_Medio"], previsoes))

                                if rmse < melhor_rmse:
                                    melhor_p, melhor_d, melhor_q = p, d, q
                                    melhor_P, melhor_D, melhor_Q = P, D, Q
                                    melhor_rmse = rmse

                            except Exception as e:
                                print(f"❌ Erro ao ajustar SARIMA({p},{d},{q})x({P},{D},{Q},{sazonalidade}): {e}")

    print(f"✅ Melhor configuração para SARIMA: (p, d, q) = ({melhor_p}, {melhor_d}, {melhor_q}) | (P, D, Q, s) = ({melhor_P}, {melhor_D}, {melhor_Q}, {sazonalidade})")

    # Ajustar o modelo final
    modelo_final = SARIMAX(
        train_data["Preco_Medio"],
        order=(melhor_p, melhor_d, melhor_q),
        seasonal_order=(melhor_P, melhor_D, melhor_Q, sazonalidade),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    resultado_final = modelo_final.fit(disp=False)

    # Fazer previsões e alinhar índice
    previsoes = resultado_final.forecast(steps=len(test_data))
    previsoes.index = test_data.index

    # Remover valores ausentes
    previsoes.dropna(inplace=True)
    test_data_validado = test_data.loc[previsoes.index]

    # Calcular métricas
    mae = mean_absolute_error(test_data_validado["Preco_Medio"], previsoes)
    rmse = np.sqrt(mean_squared_error(test_data_validado["Preco_Medio"], previsoes))
    r2 = r2_score(test_data_validado["Preco_Medio"], previsoes)

    # Salvar métricas e previsões
    helper.salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, granularidade)
    helper.salva_previsao_csv(nome_modelo.lower(), previsoes.values, granularidade)

    print(f"📊 {nome_modelo} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"📁 Previsões salvas em 'resultados/{nome_modelo.lower()}_predictions.csv'.")

def executar_sarima(caminho_teste="dados_processados", granularidade="mensal"):
    """
    Executa o pipeline completo do modelo SARIMA para a granularidade especificada.
    """
    print(f"\n🚀 Executando SARIMA com otimização de parâmetros para granularidade {granularidade}...")

    train_data, test_data = carregar_dados(caminho_teste, granularidade)

    # Tratar valores ausentes nos conjuntos de dados
    train_data, test_data = helper.tratar_nans(train_data, test_data, metodo="interpolacao")

    ajustar_sarima(train_data, test_data, "SARIMA", granularidade)

def ajustar_prophet(train_data, test_data, nome_modelo="Prophet", granularidade="mensal"):
    """
    Ajusta o modelo Prophet e realiza previsões.
    """
    if train_data is None or test_data is None:
        print("❌ Erro: Conjuntos de dados inválidos.")
        return

    # Tratar valores ausentes
    train_data, test_data = helper.tratar_nans(train_data, test_data, metodo="interpolacao")

    # Preparar os dados para o Prophet (colunas 'ds' e 'y')
    train_data_prophet = train_data.reset_index().rename(columns={"Data": "ds", "Preco_Medio": "y"})
    test_data_prophet = test_data.reset_index().rename(columns={"Data": "ds", "Preco_Medio": "y"})

    # Identificar a sazonalidade com base na granularidade
    sazonalidade = {"diaria": "daily", "semanal": "weekly", "mensal": "monthly"}.get(granularidade, "monthly")

    # Inicializar o modelo Prophet com sazonalidade
    print(f"\n🚀 Treinando o modelo {nome_modelo} com sazonalidade: {sazonalidade}...")
    modelo = Prophet()

    if sazonalidade == "weekly":
        modelo.add_seasonality(name="weekly", period=7, fourier_order=3)
    elif sazonalidade == "monthly":
        modelo.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    # Ajustar o modelo
    modelo.fit(train_data_prophet)

    # Realizar previsões
    futuro = modelo.make_future_dataframe(periods=len(test_data), freq=test_data.index.freqstr)
    previsoes = modelo.predict(futuro)

    # Ajustar previsões ao intervalo do conjunto de teste usando reindex
    previsoes = previsoes.set_index("ds").reindex(test_data_prophet["ds"])["yhat"]

    # Remover valores ausentes
    previsoes.dropna(inplace=True)
    test_data_validado = test_data.loc[previsoes.index]

    # Calcular métricas de desempenho
    mae = mean_absolute_error(test_data_validado["Preco_Medio"], previsoes)
    rmse = np.sqrt(mean_squared_error(test_data_validado["Preco_Medio"], previsoes))
    r2 = r2_score(test_data_validado["Preco_Medio"], previsoes)

    # Salvar métricas e previsões
    helper.salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, granularidade)
    helper.salva_previsao_csv(nome_modelo.lower(), previsoes.values, granularidade)

    print(f"📊 {nome_modelo} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"📁 Previsões salvas em 'resultados/{nome_modelo.lower()}_predictions.csv'.")

def executar_prophet(caminho_teste="dados_processados", granularidade="mensal"):
    """
    Executa o pipeline completo do modelo Prophet para a granularidade especificada.
    """
    print(f"\n🚀 Executando Prophet para granularidade {granularidade}...")

    train_data, test_data = carregar_dados(caminho_teste, granularidade)

    # Tratar valores ausentes nos conjuntos de dados
    train_data, test_data = helper.tratar_nans(train_data, test_data, metodo="interpolacao")

    ajustar_prophet(train_data, test_data, "Prophet", granularidade)

def diagnosticar_dados(train_data, test_data):
    print("\n📊 Diagnóstico dos Dados\n")

    # 1. Verificar dados ausentes (NaN) em treino e teste
    print("✅ Dados de Treino:")
    print(train_data.info())
    print("\nValores ausentes no treino:\n", train_data.isna().sum())
    print("\nPrimeiras linhas do treino:\n", train_data.head())

    print("\n✅ Dados de Teste:")
    print(test_data.info())
    print("\nValores ausentes no teste:\n", test_data.isna().sum())
    print("\nPrimeiras linhas do teste:\n", test_data.head())

    # 2. Checar índice e frequência
    print("\n📅 Frequência dos índices:")
    print("Treino (freq):", train_data.index.freq)
    print("Teste (freq):", test_data.index.freq)

    # Garantir consistência no índice
    if train_data.index.freq is None or test_data.index.freq is None:
        print("⚠️ Frequência não definida! Ajustando para frequência detectada...")

        # Detecta e ajusta a frequência
        train_data = train_data.asfreq(pd.infer_freq(train_data.index))
        test_data = test_data.asfreq(pd.infer_freq(test_data.index))

        print("Treino (corrigido):", train_data.index.freq)
        print("Teste (corrigido):", test_data.index.freq)

    # 3. Verificar valores infinitos
    print("\n♾️ Checando valores infinitos...")
    print("Infinitos no treino:", np.isinf(train_data).sum())
    print("Infinitos no teste:", np.isinf(test_data).sum())

    # 4. Identificar entradas duplicadas no índice
    print("\n🔍 Verificando duplicatas no índice...")
    print("Duplicatas no treino:", train_data.index.duplicated().sum())
    print("Duplicatas no teste:", test_data.index.duplicated().sum())

    return train_data, test_data
