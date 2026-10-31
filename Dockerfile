# syntax=docker/dockerfile:1
# Imagem leve: dashboard, consumer e simulador (sem TensorFlow)
FROM python:3.11-slim AS dashboard

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Usuario sem privilegios (boa pratica de seguranca de containers)
RUN groupadd --system --gid 1000 agro \
    && useradd --system --uid 1000 --gid agro --home-dir /app --shell /usr/sbin/nologin agro

COPY requirements-dashboard.txt .
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements-dashboard.txt

COPY --chown=agro:agro . .
RUN mkdir -p data_lake output logs && chown -R agro:agro data_lake output logs

USER agro
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-m", "scripts.healthcheck", "dashboard", "--url", "http://localhost:8501/_stcore/health"]

CMD ["streamlit", "run", "dashboard/app.py"]

# Imagem do modelo de visao (TensorFlow), executada sob demanda
FROM python:3.11-slim AS vision

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN groupadd --system --gid 1000 agro \
    && useradd --system --uid 1000 --gid agro --home-dir /app --shell /usr/sbin/nologin agro

COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

COPY --chown=agro:agro . .
RUN mkdir -p data_lake output logs && chown -R agro:agro data_lake output logs
USER agro

CMD ["sh", "-c", "python src/predict_folder.py --in_dir input_images --out output/results.json && python src/prepare_data.py"]

# Imagem de analise: mineracao de processos (pm4py + Graphviz), executada sob demanda
FROM python:3.11-slim AS analysis

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 agro \
    && useradd --system --uid 1000 --gid agro --home-dir /app --shell /usr/sbin/nologin agro

COPY requirements-dashboard.txt requirements-analysis.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements-analysis.txt

COPY --chown=agro:agro . .
RUN mkdir -p output logs && chown -R agro:agro output logs
USER agro

CMD ["sh", "-c", "python -m scripts.build_event_log --xes && python -m scripts.process_mining"]
