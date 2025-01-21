# Projeto de Análise de Modelos de Previsão para Séries Temporais

Este projeto realiza a análise de modelos de previsão para séries temporais utilizando o preço da gasolina no Brasil como estudo de caso. O objetivo é avaliar e comparar diferentes algoritmos de previsão, como Regressão Linear, Random Forest, ARIMA, SARIMA e Prophet.

## Estrutura do Projeto

### Diretórios Principais
- `dados_processados/`: Contém os arquivos de treino e teste gerados a partir dos dados originais.
- `metricas/`: Armazena as métricas de desempenho dos modelos.
- `visualizacao/`: Contém as funções para geração de gráficos de comparação.
- `data/` : Contém os dados que serão processados

### Arquivos Importantes
- `main.py`: Ponto de entrada do projeto, com menu interativo para execução dos modelos.
- `funcoes_menu.py`: Funções auxiliares para gerenciamento do menu e execução dos modelos.
- `modelos.py`: Contém as implementações dos modelos de previsão.
-  `dados.py`: Possui função para carregar os dados que serão utilizados.
- `requirements.txt`: Lista de dependências necessárias para rodar o projeto.
- `.gitignore`: Define os arquivos e pastas a serem ignorados pelo Git.

## Requisitos

- **Python 3.8 ou superior**
- Bibliotecas listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <PASTA_DO_PROJETO>
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Descompacte o arquivo data.zip, crie o diretório `data/` e certifique-se de que os dados originais estão disponíveis no diretório `./data/` com o padrão de nome `ca-*.csv`. 
2. Execute o arquivo principal:
   ```bash
   python main.py
   ```
3. Siga o menu interativo para:
   - Executar análises com modelos específicos
   - Exibir métricas de desempenho
   - Visualizar gráficos comparativos

## Modelos Implementados

- **Regressão Linear**: Um modelo simples baseado em relação linear entre variáveis.
- **Random Forest**: Modelo baseado em árvores de decisão.
- **ARIMA**: Modelo clássico para séries temporais estacionárias.
- **SARIMA**: Extensão sazonal do ARIMA.
- **Prophet**: Modelo desenvolvido pelo Facebook para séries temporais com forte sazonalidade.

## Estrutura de Dados

Os dados processados para treinamento e teste estão no formato:
- `train_data.csv` e `test_data.csv`
  - **Colunas**: `Data`, `Preco_Medio`, `Time_Index`

## Resultados

As métricas de desempenho para cada modelo são salvas no arquivo `metricas/resultados_modelos.csv`. Os gráficos comparativos são gerados dinamicamente com base nas previsões e valores reais.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e enviar pull requests.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
