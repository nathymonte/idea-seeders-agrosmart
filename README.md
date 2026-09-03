# AgroSmart - Reconhecimento de Folhas e Data Lake

O AgroSmart e um prototipo de agricultura inteligente para cultivo de tomates. Ele combina:

- classificacao de imagens de folhas como `Health` ou `Sick`;
- leituras simuladas de sensores IoT;
- ingestao continua via Kafka;
- ingestao historica via CSV;
- armazenamento em camadas de Data Lake;
- indicadores consolidados para apoio a decisao no dashboard Streamlit.

## Estrutura Principal

```text
idea-seeders-agrosmart/
├── config/thresholds.yaml
├── dashboard/app.py
├── data_lake/
│   ├── raw/
│   ├── trusted/
│   ├── refined/
│   └── rejected/
├── input_images/
├── models/
├── output/
├── samples/
├── scripts/
│   ├── create_data_lake.py
│   ├── historical_ingestion.py
│   ├── sensor_consumer.py
│   └── sensor_producer.py
├── services/
│   ├── data_lake.py
│   └── sensor_validator.py
├── src/
└── tests/
```

## Camadas do Data Lake

- `raw`: eventos recebidos sem transformacao e arquivos historicos preservados.
- `trusted`: leituras validadas, tipadas, padronizadas e sem duplicidade.
- `rejected`: eventos rejeitados com os motivos da rejeicao.
- `refined`: indicadores consolidados por talhao, prontos para o dashboard.

## Configuracao de Thresholds

As faixas de sensores ficam centralizadas em `config/thresholds.yaml`.

- `allowed_range`: faixa fisica aceita pela validacao. Valores fora dessa faixa vao para `rejected`.
- `expected_range`: faixa operacional esperada para nivel de atencao e recomendacoes. Campos sem `expected_range` sao validados, mas nao geram alerta operacional.

O caminho do YAML pode ser alterado com a variavel `AGROSMART_THRESHOLDS_PATH`. A configuracao e carregada uma vez por processo; se o arquivo estiver ausente ou invalido, o sistema apresenta erro claro na inicializacao.

## Instalacao Local

```bash
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

Se aparecer erro com `self.async` ao importar `kafka`, existe um pacote antigo instalado no ambiente. Corrija com:

```bash
pip uninstall -y kafka
pip install --upgrade --force-reinstall kafka-python
```

## Fluxo de Imagens

```bash
python src/predict_folder.py --in_dir input_images --out output/results.json
python src/prepare_data.py
streamlit run dashboard/app.py
```

## Fluxo de Data Lake

A ordem recomendada para validar tudo do zero e:

1. criar a estrutura do Data Lake;
2. ingerir o CSV historico;
3. subir Kafka;
4. iniciar o consumer;
5. rodar o producer para enviar eventos simulados;
6. abrir o dashboard.

Criar a estrutura:

```bash
python scripts/create_data_lake.py
```

Ingerir CSV historico:

```bash
python scripts/historical_ingestion.py --csv samples/sensor_history.csv
```

Publicar leituras simuladas no Kafka:

```bash
python src/streaming/producer.py --bootstrap localhost:29092 --count 10
```

Consumir Kafka e gravar no Data Lake:

```bash
python src/streaming/consumer.py --bootstrap localhost:29092 --refine-every 10
```

Por padrao, o consumer fica rodando continuamente, esperando novas mensagens no topico Kafka. Para testes curtos, use `--max-messages` para encerrar automaticamente depois de uma quantidade definida:

```bash
python src/streaming/consumer.py --bootstrap localhost:29092 --max-messages 5
```

Nesse caso, abra outro terminal e envie a mesma quantidade de eventos:

```bash
python src/streaming/producer.py --bootstrap localhost:29092 --count 5
```

No streaming, a camada `refined` e recalculada em micro-batches: por padrao, a cada 10 eventos validos. Eventos rejeitados nao contam para esse lote. O valor tambem pode ser configurado pela variavel `AGROSMART_REFINE_EVERY`.

Os arquivos `scripts/sensor_producer.py` e `scripts/sensor_consumer.py` continuam existindo apenas como atalhos para os modulos em `src/streaming`.

Dentro do Docker, os servicos usam `kafka:9092`. Fora do Docker, use `localhost:29092`.

## Docker

Subir dashboard e Kafka:

```bash
docker compose up --build
```

Subir tambem o consumidor de streaming:

```bash
docker compose --profile streaming up --build
```

## Testes

```bash
python -m unittest discover -s tests
```
