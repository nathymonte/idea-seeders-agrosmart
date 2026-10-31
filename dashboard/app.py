import contextlib
import json
import os
import sys
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st

from services.dashboard_data import filter_images, image_frame, sensor_frame
from services.health import overall_state, system_overview
from services.observability import get_logger, tail_logs, write_heartbeat
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
last_date = max(latest_dates) if latest_dates else datetime.now(UTC).date()
dates = st.sidebar.date_input("Periodo (UTC)", (last_date - timedelta(days=7), last_date))
predictions = st.sidebar.multiselect("Classificacao", ["Health", "Sick"], default=["Health", "Sick"])
if st.sidebar.button("Atualizar", icon=":material/refresh:"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("AgroSmart · Fase 7")


logger = get_logger("dashboard")
STATE_ICONS = {"online": "🟢", "finalizado": "🔵", "atrasado": "🟡", "offline": "🔴", "sem dados": "⚪",
               "sob demanda": "⚪"}
OVERALL = {"saudavel": ("success", "Plataforma saudavel"), "atencao": ("warning", "Plataforma em atencao"),
           "critico": ("error", "Plataforma em estado critico: consumer sem heartbeat")}


def _age(seconds):
    if seconds is None:
        return "sem registros"
    if seconds < 60:
        return f"ha {seconds:.0f} s"
    if seconds < 3600:
        return f"ha {seconds / 60:.0f} min"
    return f"ha {seconds / 3600:.1f} h"


def render_health(now):
    overview = system_overview(DATA_LAKE_PATH, now)
    kind, message = OVERALL[overall_state(overview)]
    getattr(st, kind)(f"{message} · verificado em {now:%d/%m/%Y %H:%M:%S} UTC")

    st.subheader("Componentes")
    columns = st.columns(len(overview["components"]))
    for column, item in zip(columns, overview["components"].values(), strict=True):
        status = item["status"]
        beat = status.get("heartbeat_at")
        age = max(0.0, (now - pd.Timestamp(beat).to_pydatetime()).total_seconds()) if beat else None
        column.metric(item["label"], f"{STATE_ICONS.get(item['state'], '')} {item['state']}")
        if item["state"] == "sob demanda":
            column.caption("Executa em lote quando ha novas imagens; ainda nao executado nesta instalacao")
        elif item.get("on_demand") and status.get("images") is not None:
            column.caption(f"Ultima execucao {_age(age)} · {status['images']} imagens, "
                           f"{status.get('sick_images', 0)} Sick, {status.get('associated', 0)} com talhao")
        else:
            column.caption(f"Heartbeat {_age(age)}" + (f" · estado: {status['state']}" if status.get("state") else ""))

    consumer = overview["components"]["consumer"]["status"]
    producer = overview["components"]["producer"]["status"]
    st.subheader("Indicadores de funcionamento")
    row = st.columns(4)
    row[0].metric("Eventos recebidos", consumer.get("received", 0))
    row[1].metric("Aceitos (trusted)", consumer.get("accepted", 0))
    row[2].metric("Rejeitados / duplicados", f"{consumer.get('rejected', 0)} / {consumer.get('duplicates', 0)}")
    row[3].metric("Taxa de rejeicao", f"{consumer.get('rejection_rate', 0) * 100:.1f}%")
    row = st.columns(4)
    row[0].metric("Latencia media de ingestao", f"{consumer.get('avg_ingestion_latency_s', 0):.2f} s",
                  help="Tempo entre a geracao da leitura e o recebimento pelo consumer.")
    row[1].metric("Latencia maxima", f"{consumer.get('max_ingestion_latency_s', 0):.2f} s")
    row[2].metric("Lotes consolidados", consumer.get("batches_refined", 0))
    row[3].metric("Ultimo lote (refined)", f"{consumer.get('last_batch_ms', 0):.0f} ms")
    if producer:
        st.caption(f"Simulador: {producer.get('sent', 0)} eventos enviados · cenario {producer.get('scenario')} · "
                   f"{producer.get('faults_injected', 0)} falhas injetadas · "
                   f"{producer.get('duplicates_injected', 0)} duplicatas injetadas")

    st.subheader("Data Lake")
    lake = overview["data_lake"]
    row = st.columns(4)
    for column, layer in zip(row, ("raw", "trusted", "rejected"), strict=False):
        column.metric(f"Camada {layer}", lake[layer]["records"])
        column.caption(f"Ultima escrita {_age(lake[layer]['last_write_age_s'])}")
    row[3].metric("Qualidade dos dados", "-" if lake["quality_rate"] is None else f"{lake['quality_rate'] * 100:.1f}%",
                  help="Leituras aceitas / (aceitas + rejeitadas).")
    if overview["fields"]:
        fields = pd.DataFrame(overview["fields"]).rename(columns={
            "field_id": "Talhao", "attention_level": "Nivel", "last_reading_at": "Ultima leitura",
            "updated_at": "Consolidado em", "readings_24h": "Leituras (24h)"})
        st.dataframe(fields, hide_index=True, width="stretch")

    logs = pd.DataFrame(tail_logs(2000))
    st.subheader("Atividade recente")
    if logs.empty:
        st.info("Nenhum log encontrado em logs/. Inicie o producer e o consumer para gerar atividade.")
        return
    logs["ts"] = pd.to_datetime(logs["ts"], utc=True, errors="coerce")
    if "activity" in logs:
        received = logs[logs["activity"] == "Recebida pelo consumer"]
        if not received.empty:
            per_minute = received.set_index("ts").resample("1min").size().reset_index(name="eventos")
            st.plotly_chart(px.bar(per_minute, x="ts", y="eventos", title="Vazao: eventos recebidos por minuto",
                                   labels={"ts": "Horario (UTC)"}, color_discrete_sequence=["#187b57"]),
                            width="stretch")
            if "ingestion_latency_s" in received:
                latency = received.dropna(subset=["ingestion_latency_s"])
                if not latency.empty:
                    fig = px.line(latency, x="ts", y="ingestion_latency_s", title="Latencia de ingestao por evento",
                                  labels={"ts": "Horario (UTC)", "ingestion_latency_s": "Latencia (s)"},
                                  color_discrete_sequence=["#3486b5"])
                    st.plotly_chart(fig, width="stretch")
        outcomes = logs[logs["activity"].isin(["Gravada na trusted", "Rejeitada (validacao)",
                                               "Rejeitada (payload invalido)", "Descartada (duplicada)",
                                               "Alerta emitido"])]
        if not outcomes.empty:
            counts = outcomes["activity"].value_counts().reset_index()
            counts.columns = ["Resultado", "Quantidade"]
            st.plotly_chart(px.bar(counts, x="Resultado", y="Quantidade", title="Resultado do processamento",
                                   color_discrete_sequence=["#c85057"]), width="stretch")
    levels = st.multiselect("Nivel do log", ["INFO", "WARNING", "ERROR"], default=["WARNING", "ERROR", "INFO"])
    visible = logs[logs["level"].isin(levels)].sort_values("ts", ascending=False).head(200)
    columns = [c for c in ["ts", "level", "component", "message", "case_id", "field_id", "errors"] if c in visible]
    st.dataframe(visible[columns], hide_index=True, width="stretch")
    st.caption("Logs completos em logs/<componente>.log (JSON por linha) e via `docker compose logs -f`.")


@st.fragment(run_every="5s")
def monitoring():
    if len(dates) != 2:
        st.info("Selecione o inicio e o fim do periodo.")
        return
    start = datetime.combine(dates[0], time.min, UTC)
    end = datetime.combine(dates[1], time.max, UTC)
    selected_fields = None if selected == "Todos" else {selected}
    sensors = sensor_frame(start, end, selected_fields)
    image_data = filter_images(load_images(), start, end, selected_fields)
    image_data = image_data[image_data["predicted_label"].isin(predictions)]
    current_statuses = load_statuses()
    now = datetime.now(UTC)
    with contextlib.suppress(OSError):
        write_heartbeat("dashboard", state="running", sessions_refreshed_at=now.isoformat())
    environment_tab, images_tab, data_tab, health_tab = st.tabs(["Ambiente", "Folhas", "Dados", "Saude do sistema"])
    with health_tab:
        try:
            render_health(now)
        except (OSError, ValueError, KeyError) as exc:
            logger.exception("falha na aba de saude")
            st.error(f"Nao foi possivel carregar a saude do sistema: {exc}")
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
                for column, (field, (label, unit)) in zip(st.columns(4), units.items(), strict=True):
                    column.metric(label, f"{latest.iloc[0][field]:.1f} {unit}")
            else:
                labels = {field: f"{label} ({unit})" for field, (label, unit) in units.items()}
                labels["field_id"] = "Talhao"
                st.dataframe(latest[["field_id", *units]].rename(columns=labels),
                             hide_index=True, width="stretch")
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
                              labels={"event_timestamp": "Data (UTC)", field: f"{label} ({unit})",
                                      "field_id": "Talhao"},
                              color_discrete_sequence=["#187b57", "#c85057", "#3486b5"], title=label)
                limits = config["sensor_fields"][field].get("expected_range")
                if limits:
                    fig.add_hrect(y0=limits["minimum"], y1=limits["maximum"], fillcolor="#187b57",
                                  opacity=.08, line_width=0)
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="Talhao")
                st.plotly_chart(fig, width="stretch")
            st.subheader("Medias do periodo")
            averages = sensors.groupby("field_id")[list(units)].mean().round(2)
            names = {field: f"{label} ({unit})" for field, (label, unit) in units.items()}
            st.dataframe(averages.rename(columns=names), width="stretch")
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
            verified = image_data[(image_data["label_source"] == "manifest")
                                  & image_data["actual_label"].isin(["Health", "Sick"])]
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
    logger.exception("falha ao carregar monitoramento")
    st.error(f"Nao foi possivel carregar o monitoramento: {exc}")
