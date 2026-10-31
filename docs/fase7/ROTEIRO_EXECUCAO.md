# Roteiro de execucao e coleta de evidencias - Fase 7

Execute no PowerShell, na pasta do projeto, com o Docker Desktop aberto.
Salve os prints em `docs/fase7/evidencias/` com os nomes sugeridos.

## 1. Subir a plataforma completa (execucao continua)

```powershell
docker compose --profile streaming up --build -d
docker compose ps
```

Sobe Kafka, consumer, simulador IoT continuo (3 talhoes, cenarios variados, 5% de leituras invalidas e
3% de reenvios) e dashboard. Aguarde todos os servicos ficarem `healthy` (1 a 2 minutos).

- Print: `docker compose ps` com os servicos healthy -> `01_containers_healthy.png`

## 2. Monitoramento

Abra http://localhost:8501.

- Print da aba **Saude do sistema** (componentes, indicadores, Data Lake) -> `02_saude_sistema.png`
- Print dos graficos de vazao/latencia e da tabela de logs -> `03_atividade_logs.png`
- Print da aba **Ambiente** com os 3 talhoes e recomendacoes -> `04_ambiente.png`

Logs no terminal (Ctrl+C para sair):

```powershell
docker compose logs -f --tail 20 data-lake-consumer
```

- Print dos logs JSON -> `05_logs_consumer.png`

## 3. Mineracao de processos

Deixe a plataforma rodando por pelo menos 10 minutos e execute:

```powershell
docker compose --profile analysis run --rm process-mining
```

Resultados em `output/process_mining/` (event log CSV/XES, mapas do processo, variantes, gargalos e
`analise_processo.md`).

## 4. Pipeline DevSecOps local (mesmas etapas do GitHub Actions)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ci_local.ps1
```

- Print do resumo final com as 5 etapas OK -> `06_pipeline_local.png`
- Relatorios em `artifacts/ci/`

## 5. Pipeline no GitHub Actions (opcional, recomendado)

Inclua no commit pelo menos `.github/`, `security/`, `.trivyignore`, `pyproject.toml`, `Dockerfile`,
`requirements*.txt`, `services/`, `src/`, `scripts/`, `tests/` e `dashboard/`, depois `git push`.
Na aba **Actions** do repositorio, abra a execucao "DevSecOps AgroSmart".

- Print com os 3 jobs verdes -> `07_github_actions.png`

## 6. Encerrar

```powershell
docker compose --profile streaming down
```
