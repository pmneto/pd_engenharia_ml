import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys


from kedro.io import DataCatalog


from pathlib import Path

import pandas as pd
import streamlit as st

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
import os

src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))
project_name = "engenharia_ml_kobe"
project_path = os.getcwd()
configure_project(project_name)

with KedroSession.create(project_path=project_path) as session:
    context = session.load_context()
    catalog = context.catalog

    # Carrega modelo e dados
    model = catalog.load("best_model")

    st.title("🏀 Preditor de Arremessos do Kobe - Entrada Manual")

    st.markdown("Preencha os dados do arremesso para obter a previsão:")

    # === Interface de inputs manuais (ajuste os intervalos e defaults conforme necessário)
    lat = st.number_input("Latitude", value=34.0, step=0.01)
    lon = st.number_input("Longitude", value=-118.2, step=0.01)
    period = st.selectbox("Período do jogo", [1, 2, 3, 4])
    minutes_remaining = st.number_input("Minutos restantes", min_value=0, max_value=12, value=5)
    shot_distance = st.number_input("Distância do arremesso (pés)", min_value=0, max_value=40, value=15)
    playoffs = st.selectbox("Jogo de playoff?", [0, 1])
    

    # === Monta o DataFrame com os dados inseridos
    input_data = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "minutes_remaining": minutes_remaining,
        "period": period,
        "playoffs": playoffs,
        "shot_distance": shot_distance,
    }])

    # === Previsão
    if st.button("🔮 Fazer previsão"):
        proba = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.markdown(f"### 🎯 Resultado da Previsão:")
        st.markdown(f"- **Probabilidade de Acerto:** `{proba:.2%}`")
        st.markdown(f"- **Previsão:** {'✅ Acerto' if pred == 1 else '❌ Erro'}")
