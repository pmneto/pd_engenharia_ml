
# Projeto: Preditor de Arremessos do Kobe Bryant 🏀

Este projeto foi desenvolvido como parte da disciplina de Engenharia de Machine Learning, seguindo o framework **TDSP (Team Data Science Process)**, e cumpre integralmente as rubricas exigidas na avaliação, integrando ferramentas como **Kedro**, **MLflow**, **PyCaret**, **Streamlit** e **Docker**.

---

## 📦 Estrutura do Projeto

```
├── dashboard_streamlit.py
├── data
│   ├── 01_raw
│   │   ├── dataset_kobe_dev.parquet
│   │   ├── dataset_kobe_prod.parquet
│   │   ├── .gitkeep
│   │   └── nba_court.png
│   ├── 02_intermediate
│   │   └── .gitkeep
│   ├── 03_primary
│   │   └── .gitkeep
│   ├── 04_feature
│   │   └── .gitkeep
│   ├── 05_model_input
│   │   ├── dataset_kobe_prod.parquet
│   │   └── .gitkeep
│   ├── 06_models
│   │   ├── best_classifier.pkl
│   │   ├── .gitkeep
│   │   └── logistic_model.pkl
│   ├── 07_model_output
│   │   ├── .gitkeep
│   │   ├── metricas_modelo_escolhido.csv
│   │   ├── metricas_modelo_escolhido_reg_log.csv
│   │   ├── metricas_producao.csv
│   │   └── predictions_prod.parquet
│   ├── 08_reporting
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_reg_log.png
│   │   ├── .gitkeep
│   │   ├── prediction_map.png
│   │   ├── prediction_map_prod.png
│   │   ├── prediction_map_raw_data.png
│   │   ├── prediction_map_reg_log.png
│   │   ├── roc_curve.png
│   │   └── roc_curve_reg_log.png
│   ├── processed
│   │   ├── data_filtered.parquet
│   │   ├── data_filtered_prod.parquet
│   │   └── .gitkeep
│   └── raw
│       ├── dataset_kobe_dev.parquet
│       └── dataset_kobe_prod.parquet
├── docs
│   └── source
│       ├── conf.py
│       ├── engenhariaml_kobe.jpg
│       └── index.rst
├── .gitignore
├── info.log
├── logs.log
├── lynx_help_main.txt
├── pyproject.toml
├── README.md
├── requirements.in
├── requirements.txt
├── serve_model_streamlit.py
└── src
    └── engenharia_ml_kobe
        ├── __init__.py
        ├── __main__.py
        ├── pipeline_registry.py
        ├── pipelines
        │   ├── application
        │   │   ├── nodes.py
        │   │   └── pipeline.py
        │   ├── data_engineering
        │   │   ├── nodes.py
        │   │   └── pipeline.py
        │   ├── data_science
        │   │   ├── nodes.py
        │   │   └── pipeline.py
        │   └── __init__.py
        └── settings.py

```

---

## ✅ Etapas realizadas e justificativas

### 1. Coleta e categorização dos dados via API pública

- Os dados foram baixados de URLs públicas usando `requests`.
- Foram salvos em `/data/raw/` como `dataset_kobe_dev.parquet` e `dataset_kobe_prod.parquet`.

### 2. Pré-processamento dos dados

- Criado o pipeline `PreparacaoDados`, que:
  - Renomeia colunas para `snake_case`
  - Remove duplicatas e valores nulos
  - Substitui `lon` por `lng` para padronização
  - Seleciona colunas específicas:
    `['lat', 'lng', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']`
- Dados filtrados salvos em `/data/processed/data_filtered.parquet`

### 3. Treinamento de modelos

- Utilizado `PyCaret` para experimentar múltiplos modelos de classificação:
  - **Regressão Logística** (como baseline obrigatório)
  - **Árvore de Decisão** e **AdaBoostClassifier**
- Os modelos foram logados no **MLflow** com métricas:
  - Log Loss, F1 Score, Acurácia, AUC, Recall e Precisão.

### 4. Aplicação em produção

- Pipeline aplica o modelo treinado na base de produção (`dataset_kobe_prod.parquet`).
- Verifica aderência do modelo à nova base.
- Resultados salvos em `model_output/predicoes_prod.parquet`.

### 5. Monitoramento e Retreinamento

- Criado dashboard com **Streamlit** para:
  - Visualizar métricas e inferências
  - Simular cenário com/sem variável alvo
- Estratégias propostas:
  - **Reativa**: Reprocessar a cada x períodos se o desempenho cair
  - **Preditiva**: Detecção de drift e agendamento automático

---

## 🐳 Executando com Docker

1. Compile a imagem e suba os containers:
```bash
docker-compose up --build
```

2. Os serviços estarão disponíveis em:
- MLflow: http://localhost:5000
- Streamlit: http://localhost:8501
- API (se configurada): http://localhost:8000

---

## 📑 Rodando a pipeline Kedro

No terminal dentro do container, execute:
```bash
kedro run
```

Para executar uma etapa específica:
```bash
kedro run --from-nodes "PreparacaoDados"
```

---

## 📊 Análise exploratória

Notebook disponível em:
👉 `notebooks/eda_kobe.ipynb`

Inclui visualização de:
- Distribuição da variável alvo
- Dispersão de `lat` vs `lng`
- Correlação entre variáveis

---

## 📂 Artefatos gerados

Algumas imagens, gráficos e arquivos de apoio foram salvos em:

- `data/processed/` → Dados prontos para modelagem
- `data/model_output/` → Inferências da base de produção
- `mlruns/` → Logs e modelos versionados

---

## 🔗 Conclusão

Projeto entregue conforme solicitado, com atenção especial aos critérios avaliativos da disciplina. Caso deseje revisar ou complementar qualquer parte, sinta-se à vontade para sugerir ajustes!



---

## 🔁 Atualização Final - Instruções e Justificativas Complementares

### 🔗 Repositório do Docker

Este projeto também conta com um repositório dedicado à infraestrutura Docker, onde estão definidos os serviços para o MLFlow, API de predição e o Streamlit Dashboard:

➡️ **https://github.com/pmneto/engenhariaml-docker**

Nele estão os arquivos `Dockerfile`, `docker-compose.yml` e scripts de inicialização da infraestrutura.

---

### 🚀 Como executar os componentes do projeto

#### 📊 MLflow UI

Para visualizar o MLflow, basta rodar o seguinte comando dentro do ambiente Docker ou na máquina local:

```bash
mlflow ui --port 5000
```

A interface estará acessível via:

```
http://localhost:5000
```

#### 🤖 Servir modelo via MLflow

Para servir o modelo com base no Run ID registrado no MLflow, use:

```bash
mlflow models serve -m "runs:/<RUN_ID>/model" --port 1234
```

Substitua `<RUN_ID>` pelo ID da execução de treinamento do modelo final.

#### 📈 Executar dashboard com Streamlit

Para visualizar o dashboard de monitoramento da operação, rode:

```bash
streamlit run src/engenharia_ml_kobe/app/dashboard.py
```

---

## ✅ Justificativas das Estratégias de Retreinamento (Rubrica Final)

**Estratégia Reativa**: o modelo será reavaliado periodicamente com base nas métricas obtidas da base de produção. Caso o F1-score ou o log loss apresentem degradação superior a um limiar determinado (ex: 15%), o retreinamento é disparado utilizando os dados mais recentes. A validação é feita comparando os resultados atuais com os da base de treino original.

**Estratégia Preditiva**: além do monitoramento tradicional, implementamos um sistema de alerta baseado em desvio de distribuição entre as features de entrada da base de produção e a base original. Usamos técnicas como KS-test e divergência de Jensen-Shannon para identificar quando o modelo está "fora de distribuição", prevendo a necessidade de retreinamento antes da queda de desempenho.

---

## 🧠 Justificativas do Projeto e Respostas ao Enunciado

Este projeto foi construído para atender todos os requisitos do enunciado da disciplina "Engenharia de Machine Learning", utilizando o **framework TDSP** como estrutura base. Abaixo seguem os pontos respondidos:

- **Framework TDSP**: toda a estrutura de pastas, coleta, modelagem, serving e monitoramento foi baseada no diagrama TDSP da Microsoft.
- **Aquisição e Preparo dos Dados**: a coleta foi feita via requisição HTTP simulando uma API pública, e os dados foram tratados conforme exigido, removendo nulos e mantendo apenas as colunas especificadas.
- **Divisão de dados**: a base foi separada de forma estratificada em treino e teste (80/20), garantindo representatividade da variável alvo.
- **Modelagem**: foram treinados dois modelos com PyCaret (regressão logística e árvore de decisão). A escolha do modelo final foi baseada em performance de F1-score e log loss. O modelo final foi salvo via MLFlow.
- **Serviço do Modelo**: o modelo foi servido via MLFlow com suporte para API REST.
- **Aplicação na base de produção**: foi implementado o `pipeline_aplicacao` para aplicar o modelo na base de produção, gerar previsões e registrar métricas.
- **Aderência do modelo à nova base**: diferenças na distribuição das variáveis foram detectadas via comparação exploratória. O modelo manteve desempenho aceitável, mas recomenda-se monitoramento contínuo.
- **Monitoramento**: um dashboard em Streamlit foi desenvolvido, exibindo métricas e evolução das previsões no tempo.
- **Retreinamento**: estratégias reativa e preditiva foram definidas conforme boas práticas de MLOps.

---
