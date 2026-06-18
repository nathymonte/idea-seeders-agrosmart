FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "python src/predict_folder.py --in_dir input_images --out output/results.json && python src/prepare_data.py && streamlit run dashboard/app.py"]