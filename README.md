
# Projeto: Preditor de Arremessos do Kobe Bryant 🏀

Este projeto foi desenvolvido como parte da disciplina de Engenharia de Machine Learning, seguindo o framework **TDSP (Team Data Science Process)**, e segue integralmente as rubricas exigidas na avaliação, integrando ferramentas como **Kedro**, **MLflow**, **PyCaret**, **Streamlit** e **Docker**.

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
- Foram salvos em `/data/01_raw/` como `dataset_kobe_dev.parquet` e `dataset_kobe_prod.parquet`.

### 2. Pré-processamento dos dados

- Criado o pipeline `PreparacaoDados`, que:
  - Renomeia colunas para `snake_case`
  - Remove duplicatas e valores nulos
  - Substitui `lon` por `lng` para padronização
  - Seleciona colunas específicas:
    `['lat', 'lng', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']`
- Dados filtrados salvos em `/data/processed/data_filtered.parquet` 

**Obs.: o diretório data/processed foi levado em conta estritamente por estar escrito e solicitado desta maneira no enunciado.**

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
- Streamlit (dashboard): http://localhost:8501
- Streamlit (Serve Model UI): http://localhost:8502
- MLflow (API para servir o Modelo): http://localhost:5000/<RUN_ID>/model
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

Projeto entregue conforme solicitado, com atenção especial aos critérios avaliativos da disciplina. 



---

## 🔁Instruções e Justificativas Complementares

### 🔗 Repositório do Docker

Este projeto também conta com um repositório dedicado à infraestrutura Docker, onde estão definidos os serviços para o MLFlow, API de predição e o Streamlit Dashboard:

➡️ **https://github.com/pmneto/engenhariaml-docker**

Nele estão os arquivos `Dockerfile`, `docker-compose.yml` e scripts de inicialização da infraestrutura.

Esta solicitação foi feita em aula pelo tutor onde por uma quantidade bastante grande de diferenças de ambientes e dificuldades em importação de dependencias, foi sugerido/solicitado que fosse disponibilizado em docker o projeto.
---

### 🚀 Como executar os componentes do projeto

#### 📊 MLflow UI

Para visualizar o MLflow, basta rodar o seguinte comando dentro do ambiente Docker ou na máquina local:

```bash
mlflow ui --port 5000
```

OU 

```bash
mlflow ui
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
streamlit run dashboard_streamlit.py
```
Para visualizar uma interface de usuário para o projeto:

```bash
streamlit run serve_model_streamlit.py
```
---

## ✅ Justificativas das Estratégias de Retreinamento (Rubrica 4.4)

Embora este projeto tenha sido desenvolvido como um trabalho final de disciplina, e portanto não implemente efetivamente um ciclo de retreinamento em produção, é importante refletir sobre como esse processo deveria acontecer em um ambiente real de MLOps.


📌 Estratégia Reativa
Uma abordagem reativa seria baseada no monitoramento contínuo de métricas de desempenho do modelo, como o F1-score e o Log Loss, aplicadas diretamente sobre a base de produção. Esse monitoramento pode ser feito através do próprio MLflow UI, que registra as execuções de inferência e permite comparar resultados de versões anteriores com a atual.

Por exemplo, se o modelo que está em operação começar a apresentar uma degradação nas métricas (ex: F1-score caindo mais de 15% em relação ao que foi observado na base de teste), isso pode servir como gatilho para retreinar o modelo com dados mais recentes.

📌 Estratégia Preditiva
Já a abordagem preditiva seria implementada através de análises estatísticas das distribuições dos dados de entrada, comparando os dados de produção com os dados originais usados para treinar o modelo. Técnicas como KS-Test ou Divergência de Jensen-Shannon podem ajudar a detectar quando as variáveis (ex: shot_distance, minutes_remaining, etc.) saem do padrão esperado, indicando um possível "drift" nos dados.

Neste caso, mesmo que o desempenho ainda não tenha caído significativamente, o sistema já seria capaz de prever que o modelo pode vir a se degradar, permitindo um retreinamento proativo.

---
