from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AgroSmart Dashboard", layout="wide")

CSV_PATH = Path("output/classificacoes.csv")

st.title("AgroSmart - Painel de Monitoramento de Folhas")
st.write("Dashboard interativo com resultados simulados de classificação de folhas.")

if not CSV_PATH.exists():
    st.error("Arquivo output/classificacoes.csv não encontrado. Rode antes: python src/prepare_data.py")
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