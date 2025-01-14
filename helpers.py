import pandas as pd
def salvar_metricas_em_csv(nome_modelo, mae, rmse, r2, arquivo_resultados="resultados_modelos.csv"):
    # Criar um DataFrame com as métricas
    df_resultado = pd.DataFrame({
        "Modelo": [nome_modelo],
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2]
    })

    # Salvar o DataFrame no arquivo CSV, no modo 'append'
    df_resultado.to_csv(arquivo_resultados, mode='a', index=False, header=not pd.io.common.file_exists(arquivo_resultados))
    print(f"Resultados do {nome_modelo} adicionados a '{arquivo_resultados}'")

def salva_teste_previsao_csv(nome_modelo, test_data, predictions):
    # Salvar o conjunto de teste e previsões em CSV para análise posterior
    test_data.to_csv(f"{nome_modelo}_test_data.csv", index=False)
    pd.DataFrame({"Data": test_data['Data'], "Previsao": predictions}).to_csv(f"{nome_modelo}_predictions.csv", index=False)
