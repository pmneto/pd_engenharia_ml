# engenharia_ml_kobe

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## 🧭 Contexto Geral

Este projeto segue a estrutura do framework **Kedro**, mas foi desenvolvido de acordo com as diretrizes do **Microsoft TDSP (Team Data Science Process)**, conforme solicitado no enunciado da atividade. Todos os artefatos gerados, pipelines e registros de experimentos são compatíveis com as fases previstas no ciclo de vida TDSP (ingestão, preparação, modelagem, scoring e monitoramento).

---

## ✅ Status do Projeto

- ✅ Ingestão e limpeza dos dados implementada
- ✅ Separação de treino/teste com estratificação
- ✅ Registro no MLflow do pipeline de preparação
- ✅ Treinamento de dois modelos (Regressão Logística e Classificador)
- ✅ Registro manual de parâmetros e métricas no MLflow
- 🔜 Registro do modelo final e deploy via aplicação
- 🔜 Avaliação do modelo com base de produção
- 🔜 Dashboard de monitoramento via Streamlit

---

## Como instalar as dependências

Declare as dependências no arquivo `requirements.txt`.

Para instalar, execute:

```
pip install -r requirements.txt
```

## Como rodar o pipeline do Kedro

Execute o pipeline com:

```
kedro run
```

Ou rode um node específico:

```
kedro run --from-nodes "nome_do_node"
```

## Como rodar o MLflow

Inicie o servidor MLflow local com:

```
mlflow ui
```

Depois, acesse: http://127.0.0.1:5000

---

## Organização por Fases (TDSP)

| Fase TDSP         | Implementação no Projeto                            |
|-------------------|-----------------------------------------------------|
| Ingestão          | `download_data` / dados em `/data/raw`             |
| Preparação        | `preprocess_data` → `/data/processed`              |
| Modelagem         | `train_models_node` com PyCaret e MLflow           |
| Avaliação         | Métricas logadas no MLflow                         |
| Operacionalização | A definir: aplicação + Streamlit + scoring final   |
| Monitoramento     | Planejado para o dashboard                         |

---

## Como testar o projeto

Veja o arquivo `src/tests/test_run.py` para instruções.

Execute os testes com:

```
pytest
```

Você pode configurar o limite mínimo de cobertura em `pyproject.toml`, na seção `[tool.coverage.report]`.

---

## Como trabalhar com notebooks no Kedro

Use `kedro jupyter` ou `kedro ipython` para acessar os objetos `context`, `session`, `catalog` e `pipelines` já carregados.

### Jupyter Notebook
```
kedro jupyter notebook
```

### JupyterLab
```
kedro jupyter lab
```

### IPython
```
kedro ipython
```

### Ignorar saídas de notebook no Git
Para remover as saídas de células antes de commitar:
```
nbstripout --install
```

---

## Empacotamento do Projeto

Veja a [documentação oficial](https://docs.kedro.org/en/stable/tutorial/package_a_project.html) para empacotar o projeto como biblioteca ou gerar documentação.

