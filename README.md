
# Projeto: Preditor de Arremessos do Kobe Bryant 🏀

Este projeto foi desenvolvido como parte da disciplina de Engenharia de Machine Learning, seguindo o framework **TDSP (Team Data Science Process)**, e cumpre integralmente as rubricas exigidas na avaliação, integrando ferramentas como **Kedro**, **MLflow**, **PyCaret**, **Streamlit** e **Docker**.

---

## 📦 Estrutura do Projeto

```
engenharia_ml_kobe/
├── data/
│   ├── raw/                  # Dados brutos de desenvolvimento e produção
│   ├── processed/            # Dados pós-processamento (filtrados)
│   └── model_output/         # Métricas, gráficos e inferências
├── notebooks/
│   └── eda_kobe.ipynb        # Análise exploratória dos dados
├── conf/base/                # Catálogo Kedro + parâmetros de configuração
├── src/
│   └── engenharia_ml_kobe/   # Pipelines, nodes e pipelines de aplicação
├── README.md
└── docker-compose.yml
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

