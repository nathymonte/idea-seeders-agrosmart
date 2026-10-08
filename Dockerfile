FROM python:3.11-slim AS dashboard

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

COPY requirements-dashboard.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-dashboard.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py"]

FROM python:3.11-slim AS vision
WORKDIR /app
COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "python src/predict_folder.py --in_dir input_images --out output/results.json && python src/prepare_data.py"]
