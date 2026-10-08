import json
import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st

from services.dashboard_data import sensor_frame, image_frame, filter_images
from services.paths import DATA_LAKE_PATH, IMAGE_CSV_PATH, PROJECT_ROOT
from services.threshold_config import get_threshold_config

st.set_page_config(page_title="AgroSmart", page_icon=":material/eco:", layout="wide")
st.markdown("""<style>
.stApp {background:#f7f9f8; color:#172b25}
[data-testid=stMetricValue] {font-size:1.65rem}
[data-testid=stMetricLabel] {white-space:normal}
.block-container {padding-top:2rem; padding-bottom:2rem}
@media(max-width:640px) {[data-testid=stHorizontalBlock] {flex-wrap:wrap}}
</style>""", unsafe_allow_html=True)
st.title("AgroSmart")
st.caption("Monitoramento de talhoes · Tomate")


@st.cache_data(ttl=5)
def load_statuses():
    statuses = {}
    for path in (DATA_LAKE_PATH / "refined/field_status").glob("*.json"):
        statuses[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return statuses


@st.cache_data(ttl=5)
def load_images():
    return image_frame(IMAGE_CSV_PATH)


statuses = load_statuses()
images = load_images()
fields = sorted(set(statuses) | set(images["field_id"]) - {""})
st.sidebar.title("Monitoramento")
selected = st.sidebar.selectbox("Talhao", ["Todos"] + fields)
latest_dates = [pd.Timestamp(s["sensor_summary"]["last_reading_at"]).date() for s in statuses.values()]
if images["data_analise"].notna().any():
    latest_dates.append(images["data_analise"].max().date())
last_date = max(latest_dates) if latest_dates else datetime.now(timezone.utc).date()
dates = st.sidebar.date_input("Periodo (UTC)", (last_date - timedelta(days=7), last_date))
predictions = st.sidebar.multiselect("Classificacao", ["Health", "Sick"], default=["Health", "Sick"])
if st.sidebar.button("Atualizar", icon=":material/refresh:"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("AgroSmart · Fase 6")


@st.fragment(run_every="5s")
def monitoring():
    if len(dates) != 2:
        st.info("Selecione o inicio e o fim do periodo.")
        return
    start = datetime.combine(dates[0], time.min, timezone.utc)
    end = datetime.combine(dates[1], time.max, timezone.utc)
    selected_fields = None if selected == "Todos" else {selected}
    sensors = sensor_frame(start, end, selected_fields)
    image_data = filter_images(load_images(), start, end, selected_fields)
    image_data = image_data[image_data["predicted_label"].isin(predictions)]
    current_statuses = load_statuses()
    now = datetime.now(timezone.utc)
    environment_tab, images_tab, data_tab = st.tabs(["Ambiente", "Folhas", "Dados"])
    with environment_tab:
        if sensors.empty:
            st.info("Sem leituras ambientais neste periodo.")
        else:
            latest = sensors.groupby("field_id", as_index=False).tail(1)
            st.subheader("Ultimas leituras")
            units = {"soil_moisture_percent": ("Umidade do solo", "%"),
                     "air_temperature_celsius": ("Temperatura", "°C"),
                     "air_humidity_percent": ("Umidade do ar", "%"),
                     "luminosity_lux": ("Luminosidade", "lux")}
            if len(latest) == 1:
                for column, (field, (label, unit)) in zip(st.columns(4), units.items()):
                    column.metric(label, f"{latest.iloc[0][field]:.1f} {unit}")
            else:
                labels = {field: f"{label} ({unit})" for field, (label, unit) in units.items()}
                labels["field_id"] = "Talhao"
                st.dataframe(latest[["field_id"] + list(units)].rename(columns=labels), hide_index=True, width="stretch")
            last_reading = sensors["event_timestamp"].max().to_pydatetime()
            st.caption(f"Ultima observacao: {last_reading:%d/%m/%Y %H:%M:%S} UTC · {len(sensors)} leituras no periodo")
            stale_seconds = float(os.getenv("AGROSMART_STALE_SECONDS", "300"))
            stale_fields = [row["field_id"] for _, row in latest.iterrows()
                            if (now - row["event_timestamp"].to_pydatetime()).total_seconds() > stale_seconds]
            if stale_fields:
                st.warning("Sem dados recentes: " + ", ".join(stale_fields))
            st.subheader("Situacao operacional")
            levels = {"normal": "Normal", "medium": "Atencao", "high": "Alta atencao"}
            for field in sorted(sensors["field_id"].unique()):
                status = current_statuses.get(field, {})
                st.markdown(f"**{field} · {levels.get(status.get('attention_level'), 'Sem consolidacao')}**")
                st.caption("Ultima consolidacao operacional, independente do filtro historico")
                if status.get("decision_basis") != "latest_reading":
                    st.warning("Consolidacao anterior a Fase 6; indicadores aguardam reprocessamento.")
                else:
                    refined_time = pd.Timestamp(status["sensor_summary"]["last_reading_at"])
                    if refined_time < sensors[sensors["field_id"] == field]["event_timestamp"].max():
                        st.warning("Novas leituras aguardam consolidacao.")
                    st.info(status.get("recommendation", "Sem recomendacao"))
                    for reason in status.get("decision_reasons", []):
                        st.caption(reason)
                    st.caption(f"Consolidado em {status.get('updated_at', '')}")
            st.subheader("Evolucao ambiental")
            config = get_threshold_config()
            for field, (label, unit) in units.items():
                fig = px.line(sensors, x="event_timestamp", y=field, color="field_id", markers=True,
                              labels={"event_timestamp": "Data (UTC)", field: f"{label} ({unit})", "field_id": "Talhao"},
                              color_discrete_sequence=["#187b57", "#c85057", "#3486b5"], title=label)
                limits = config["sensor_fields"][field].get("expected_range")
                if limits:
                    fig.add_hrect(y0=limits["minimum"], y1=limits["maximum"], fillcolor="#187b57", opacity=.08, line_width=0)
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="Talhao")
                st.plotly_chart(fig, width="stretch")
            st.subheader("Medias do periodo")
            averages = sensors.groupby("field_id")[list(units)].mean().round(2)
            st.dataframe(averages.rename(columns={field: f"{label} ({unit})" for field, (label, unit) in units.items()}), width="stretch")
    with images_tab:
        if image_data.empty:
            st.info("Sem analises de folhas neste periodo e talhao.")
        else:
            columns = st.columns(4)
            columns[0].metric("Imagens", len(image_data))
            columns[1].metric("Saudaveis", int((image_data["predicted_label"] == "Health").sum()))
            columns[2].metric("Indicios de doenca", int((image_data["predicted_label"] == "Sick").sum()))
            columns[3].metric("Confianca media", f"{image_data['confidence'].mean():.1f}%")
            if image_data["metadata_source"].isin(["legacy_simulated", "simulation"]).any():
                st.warning("Metadados de demonstracao simulados presentes neste conjunto.")
            if (image_data["field_id"] == "").any():
                st.warning("Imagens sem talhao associado nao participam das decisoes por talhao.")
            st.caption("Classificacao binaria Health/Sick. Doencas especificas nao diagnosticadas.")
            fig = px.pie(image_data, names="predicted_label_pt", title="Classificacao de folhas",
                         color_discrete_sequence=["#187b57", "#c85057"])
            st.plotly_chart(fig, width="stretch")
            trend = image_data.groupby(image_data["data_analise"].dt.date).size().reset_index(name="quantidade")
            st.plotly_chart(px.line(trend, x="data_analise", y="quantidade", markers=True,
                                   title="Analises de folhas por dia"), width="stretch")
            by_field = image_data.assign(field_id=image_data["field_id"].replace("", "Sem associacao"))
            grouped = by_field.groupby(["field_id", "predicted_label_pt"]).size().reset_index(name="quantidade")
            st.plotly_chart(px.bar(grouped, x="field_id", y="quantidade", color="predicted_label_pt",
                                  barmode="group", title="Classificacoes por talhao"), width="stretch")
            verified = image_data[(image_data["label_source"] == "manifest") & image_data["actual_label"].isin(["Health", "Sick"])]
            if not verified.empty:
                correct = verified["is_correct"].astype(str).str.lower().eq("true")
                st.metric("Acerto nas imagens com rotulo de referencia", f"{correct.mean() * 100:.1f}%")
            image_choice = st.selectbox("Imagem analisada", image_data.index,
                                       format_func=lambda i: str(image_data.loc[i, "image_name"]))
            row = image_data.loc[image_choice]
            image_root = PROJECT_ROOT / "input_images"
            image_path = (image_root / (row["image_path"] or row["image_name"])).resolve()
            if image_path.is_relative_to(image_root.resolve()) and image_path.is_file():
                st.image(str(image_path), caption=f"{row['predicted_label_pt']} · {row['confidence']:.1f}%", width=280)
            st.dataframe(image_data.drop(columns=["is_correct"]), hide_index=True, width="stretch")
    with data_tab:
        st.subheader("Leituras processadas")
        st.dataframe(sensors, hide_index=True, width="stretch")
        st.download_button("Exportar leituras", sensors.to_csv(index=False).encode("utf-8-sig"),
                           "leituras_processadas.csv", "text/csv", icon=":material/download:")
        st.subheader("Analises de folhas")
        st.dataframe(image_data, hide_index=True, width="stretch")
        st.download_button("Exportar analises", image_data.to_csv(index=False).encode("utf-8-sig"),
                           "analises_folhas.csv", "text/csv", icon=":material/download:")


try:
    monitoring()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Nao foi possivel carregar o monitoramento: {exc}")
