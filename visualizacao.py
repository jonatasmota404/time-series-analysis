import matplotlib.pyplot as plt
import pandas as pd

def plot_comparacao_modelo(nome_modelo, caminho_test_data, caminho_predictions):
    try:
        df_test = pd.read_csv(caminho_test_data)
        df_predictions = pd.read_csv(caminho_predictions)

        if len(df_predictions) != len(df_test):
            raise ValueError(
                f"O número de previsões ({len(df_predictions)}) não corresponde ao número de valores reais ({len(df_test)})."
            )

        df_predictions["Data"] = pd.to_datetime(df_predictions["Data"])

        plt.figure(figsize=(10, 6))
        plt.plot(df_predictions["Data"], df_predictions["Preco_Real"], label="Valor Real", color="blue")
        plt.plot(df_predictions["Data"], df_predictions["Previsao"], label="Previsão", color="orange", linestyle="--")
        plt.title(f"Comparação de Previsões - {nome_modelo}")
        plt.xlabel("Data")
        plt.ylabel("Preço Médio")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except ValueError as e:
        print(f"Erro: {e}")
