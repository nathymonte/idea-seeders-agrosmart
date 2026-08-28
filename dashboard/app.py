import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AgroSmart Dashboard", layout="wide")

CSV_PATH = Path("output/classificacoes.csv")
FIELD_STATUS_DIR = Path("data_lake/refined/field_status")

st.title("AgroSmart - Painel de Monitoramento")
st.write("Dashboard com classificacoes de folhas e indicadores consolidados do Data Lake.")

field_status_files = sorted(FIELD_STATUS_DIR.glob("*.json")) if FIELD_STATUS_DIR.exists() else []
if field_status_files:
    st.subheader("Status atual do talhao")
    selected_status = st.selectbox(
        "Talhao",
        options=field_status_files,
        format_func=lambda path: path.stem,
    )
    with selected_status.open("r", encoding="utf-8") as file:
        field_status = json.load(file)

    sensor_summary = field_status.get("sensor_summary", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nivel de atencao", field_status.get("attention_level", "N/A"))
    c2.metric("Umidade solo media", f"{sensor_summary.get('average_soil_moisture_percent', 0):.1f}%")
    c3.metric("Temperatura media", f"{sensor_summary.get('average_air_temperature_celsius', 0):.1f} C")
    c4.metric("Umidade ar media", f"{sensor_summary.get('average_air_humidity_percent', 0):.1f}%")
    c5.metric("Leituras", sensor_summary.get("readings_count", 0))

    st.info(field_status.get("recommendation", "Sem recomendacao gerada."))
    with st.expander("Detalhes do Data Lake"):
        st.json(field_status)
else:
    st.warning("Status refinado nao encontrado. Rode: python scripts/historical_ingestion.py")

if not CSV_PATH.exists():
    st.error("Arquivo output/classificacoes.csv nao encontrado. Rode antes: python src/prepare_data.py")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Ajustes
df["data_analise"] = pd.to_datetime(df["data_analise"])
df["is_correct_text"] = df["is_correct"].map({True: "Acerto", False: "Erro"})

# Sidebar
st.sidebar.header("Filtros")
localidades = st.sidebar.multiselect(
    "Localidade",
    options=sorted(df["localidade"].unique()),
    default=sorted(df["localidade"].unique())
)

status = st.sidebar.multiselect(
    "Status previsto",
    options=sorted(df["predicted_label_pt"].unique()),
    default=sorted(df["predicted_label_pt"].unique())
)

anomalias = st.sidebar.multiselect(
    "Anomalia",
    options=sorted(df["anomalia"].unique()),
    default=sorted(df["anomalia"].unique())
)

df_filtrado = df[
    (df["localidade"].isin(localidades)) &
    (df["predicted_label_pt"].isin(status)) &
    (df["anomalia"].isin(anomalias))
]

# KPIs
total = len(df_filtrado)
saudaveis = (df_filtrado["predicted_label"] == "Health").sum()
doentes = (df_filtrado["predicted_label"] == "Sick").sum()
taxa_acerto = (df_filtrado["is_correct"].mean() * 100) if total > 0 else 0
confianca_media = df_filtrado["confidence"].mean() if total > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Imagens analisadas", total)
c2.metric("Saudáveis", saudaveis)
c3.metric("Doentes", doentes)
c4.metric("Taxa de acerto", f"{taxa_acerto:.1f}%")
c5.metric("Confiança média", f"{confianca_media:.1f}%")

# Gráfico 1 - saudável vs doente
fig_status = px.pie(
    df_filtrado,
    names="predicted_label_pt",
    title="Distribuição de folhas por status previsto"
)
st.plotly_chart(fig_status, use_container_width=True)

# Gráfico 2 - anomalias
anomalia_count = df_filtrado["anomalia"].value_counts().reset_index()
anomalia_count.columns = ["anomalia", "quantidade"]
fig_anomalia = px.bar(
    anomalia_count,
    x="anomalia",
    y="quantidade",
    title="Frequência por anomalia"
)
st.plotly_chart(fig_anomalia, use_container_width=True)

# Gráfico 3 - tendência por data
por_data = df_filtrado.groupby("data_analise").size().reset_index(name="quantidade")
fig_data = px.line(
    por_data,
    x="data_analise",
    y="quantidade",
    markers=True,
    title="Tendência de análises por período"
)
st.plotly_chart(fig_data, use_container_width=True)

# Gráfico 4 - localidade
por_localidade = df_filtrado.groupby(["localidade", "predicted_label_pt"]).size().reset_index(name="quantidade")
fig_local = px.bar(
    por_localidade,
    x="localidade",
    y="quantidade",
    color="predicted_label_pt",
    barmode="group",
    title="Distribuição por localidade"
)
st.plotly_chart(fig_local, use_container_width=True)

# Tabela detalhada
st.subheader("Tabela detalhada")
st.dataframe(
    df_filtrado.sort_values(by="data_analise"),
    use_container_width=True
)

# Erros do modelo
st.subheader("Casos com erro de classificação")
erros = df_filtrado[~df_filtrado["is_correct"]]
if erros.empty:
    st.success("Nenhum erro encontrado nos filtros aplicados.")
else:
    st.dataframe(erros, use_container_width=True)
