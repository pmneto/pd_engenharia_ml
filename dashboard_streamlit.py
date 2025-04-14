import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Monitoramento - Modelo Kobe")

# Carrega as métricas
metricas_path = "data/07_model_output/metricas_producao_dt.csv"
df = pd.read_csv(metricas_path)

# Mostra tabela geral
st.subheader("📁 Métricas de Execuções")
st.dataframe(df.tail(10))

# Métricas agregadas
st.subheader("📈 Métricas ao Longo do Tempo")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Último F1", f"{df['f1_score'].iloc[-1]:.2f}")
    st.line_chart(df["f1_score"])

with col2:
    st.metric("Última Acurácia", f"{df['accuracy'].iloc[-1]:.2f}")
    st.line_chart(df["accuracy"])

with col3:
    st.metric("Último Log Loss", f"{df['log_loss'].iloc[-1]:.4f}")
    st.line_chart(df["log_loss"])

# Distribuição das previsões (se salva os labels)
if "prediction_label" in df.columns:
    st.subheader("📊 Distribuição das Previsões")
    st.bar_chart(df["prediction_label"].value_counts(normalize=True))

# Última execução
st.subheader("📅 Última Execução")
st.write(f"Rodada em: {df['timestamp'].iloc[-1]}")
