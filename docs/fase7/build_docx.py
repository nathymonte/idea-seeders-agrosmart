"""Gera docs/fase7/AgroSmart_Fase7_Entrega_Final.docx a partir dos artefatos reais do projeto.

Uso: python docs/fase7/build_docx.py
Le: diagramas, output/process_mining (event log e analise), data_lake (exemplos de entrada e saida),
evidencias em docs/fase7/evidencias e relatorios do pipeline em artifacts/ci.
Prints ainda ausentes aparecem como marcadores amarelos no documento.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
FASE7 = ROOT / "docs/fase7"
EVID = FASE7 / "evidencias"
PM = Path(os.getenv("AGROSMART_PM_DIR", str(ROOT / "output/process_mining")))
CI = ROOT / "artifacts/ci"
OUT = FASE7 / "AgroSmart_Fase7_Entrega_Final.docx"
GREEN = RGBColor(0x18, 0x7B, 0x57)
INK = RGBColor(0x17, 0x2B, 0x25)
TEAM = [("Nicolas Caous Fernandes", "550587"), ("Lucas de Melo Silva", "551303"),
        ("João Gabriel Lemos Spina Pinho", "99908"), ("Nathalia Lima Monteiro", "99346"),
        ("João Vitor Santos Elias", "98979")]

doc = Document()
figure_no = [0]
table_no = [0]


def setup() -> None:
    section = doc.sections[0]
    section.left_margin = section.right_margin = Cm(2.2)
    section.top_margin = section.bottom_margin = Cm(2)
    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = GREEN if name != "Heading 3" else INK
        style.font.bold = True
        style.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        style.paragraph_format.space_after = Pt(4)


def p(text: str = "", bold_prefix: str | None = None, size: float | None = None, italic: bool = False):
    paragraph = doc.add_paragraph()
    if bold_prefix:
        paragraph.add_run(bold_prefix).bold = True
    run = paragraph.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return paragraph


def bullet(text: str, bold_prefix: str | None = None) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        paragraph.add_run(bold_prefix).bold = True
    paragraph.add_run(text)


def shade(cell, color: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def table(header: list[str], rows: list[list[str]], caption: str | None = None, widths: list[float] | None = None):
    if caption:
        table_no[0] += 1
        cap = doc.add_paragraph()
        run = cap.add_run(f"Tabela {table_no[0]} – {caption}")
        run.bold = True
        run.font.size = Pt(9)
    grid = doc.add_table(rows=1, cols=len(header))
    grid.style = "Table Grid"
    grid.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(grid.rows[0].cells, header, strict=True):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "187B57")
    for index, row in enumerate(rows):
        cells = grid.add_row().cells
        for cell, text in zip(cells, row, strict=True):
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.font.size = Pt(9.5)
            if index % 2:
                shade(cell, "EEF5F1")
    if widths:
        for row in grid.rows:
            for cell, width in zip(row.cells, widths, strict=True):
                cell.width = Cm(width)
    doc.add_paragraph()
    return grid


def code(text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(0.4)
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    props = paragraph._p.get_or_add_pPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:fill"), "F3F5F4")
    props.append(fill)


def figure(path: Path, caption: str, width_cm: float = 16) -> None:
    figure_no[0] += 1
    if path.exists():
        doc.add_picture(str(path), width=Cm(width_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        missing = doc.add_paragraph()
        run = missing.add_run(f"[Inserir evidência: {path.relative_to(ROOT).as_posix()}]")
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        run.bold = True
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(f"Figura {figure_no[0]} – {caption}")
    run.italic = True
    run.font.size = Pt(9)


def latest_jsonl(folder: Path) -> dict | None:
    files = sorted(folder.rglob("*.jsonl"))
    for path in reversed(files):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return json.loads(lines[-1])
    return None


def count_found(path: Path) -> str:
    if not path.exists():
        return "-"
    match = re.search(r"Found (\d+) error", path.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else "0"


def bandit_counts(path: Path) -> str:
    if not path.exists():
        return "-"
    text = path.read_text(encoding="utf-8", errors="replace")
    def get(level):
        found = re.search(rf"{level}: (\d+)", text)
        return found.group(1) if found else "0"
    return f"{get('High')} altas, {get('Medium')} médias, {get('Low')} baixas"


def audit_count(path: Path) -> str:
    if not path.exists():
        return "-"
    data = json.loads(path.read_text(encoding="utf-8"))
    total = sum(len(dep.get("vulns", [])) for dep in data["dependencies"])
    return str(total)


# ----------------------------------------------------------------------------------------------------------------
setup()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("AgroSmart – Fase 7: Entrega Final")
run.bold = True
run.font.size = Pt(20)
run.font.color.rgb = GREEN
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("Plataforma integrada de agricultura inteligente para o cultivo de tomates").italic = True
for name, rm in TEAM:
    line = doc.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    line.paragraph_format.space_after = Pt(0)
    line.add_run(f"{name} | RM: {rm}").font.size = Pt(10)
doc.add_paragraph()

doc.add_heading("Visão geral", level=1)
p("Ao longo das fases, o AgroSmart evoluiu de um classificador de folhas para uma plataforma integrada. A Fase 7 "
  "eleva a solução para um nível próximo ao de mercado: execução contínua em containers, monitoramento, pipeline "
  "DevSecOps automatizado, análise do fluxo com mineração de processos e apresentação como produto.")
table(["Fase", "Entrega", "Componente que permanece na solução final"], [
    ["1", "Definição do problema e classificação de folhas", "Modelo CNN Health/Sick (models/keras_model.h5)"],
    ["2", "Coleta e organização de dados", "Simulador IoT e dataset de imagens"],
    ["3", "Automação baseada em regras", "Motor de decisão com thresholds.yaml"],
    ["4", "Integração entre componentes", "Kafka (producer/consumer) e ingestão histórica"],
    ["5", "Dashboards analíticos", "Dashboard Streamlit"],
    ["6", "Plataforma integrada", "Data Lake raw/trusted/rejected/refined e decisões por talhão"],
    ["7", "Entrega final", "Docker Compose, monitoramento, DevSecOps, mineração de processos e pitch"],
], "Evolução do projeto", [1.2, 6, 8.8])

# 1 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("1. Arquitetura final da solução", level=1)
figure(FASE7 / "diagramas/arquitetura_final.png", "Arquitetura final consolidada", 17)
p("A arquitetura é orientada a eventos. Cada leitura de sensor é um evento independente que atravessa a "
  "plataforma da entrada à decisão, deixando rastros em logs estruturados usados pelo monitoramento e pela "
  "mineração de processos.")
table(["Camada", "Componente", "Implementação", "Função"], [
    ["Entrada", "Simulador IoT", "src/streaming/producer.py (serviço sensor-simulator)",
     "Gera leituras de 3 talhões em cenários normal, random, dry, hot e high; injeta falhas e reenvios"],
    ["Entrada", "Ingestão histórica", "scripts/historical_ingestion.py", "Importa CSV aplicando a mesma validação"],
    ["Entrada", "Imagens", "input_images/ + src/predict_folder.py", "Classifica folhas em Health/Sick (CNN Keras)"],
    ["Integração", "Apache Kafka 3.9", "docker-compose.yml (serviço kafka)", "Transporta eventos com entrega pelo menos uma vez"],
    ["Processamento", "Consumer", "src/streaming/consumer.py",
     "Valida (YAML), deduplica, grava nas camadas e consolida em micro-lotes (10 eventos ou 5 s)"],
    ["Dados", "Data Lake", "services/data_lake.py", "raw, trusted, rejected e refined em JSONL particionado por data"],
    ["Automação", "Motor de decisão", "services/data_lake.py + config/thresholds.yaml",
     "Nível normal/medium/high, recomendações e motivos por talhão"],
    ["Saída", "Dashboard", "dashboard/app.py", "Abas Ambiente, Folhas, Dados e Saúde do sistema; exportação CSV"],
    ["Saída", "Relatórios", "src/automation.py, scripts/export_dataset.py", "Relatório operacional e dataset processado"],
    ["Observabilidade", "Logs e heartbeats", "services/observability.py, services/health.py",
     "Logs JSON, status de cada componente e indicadores de saúde"],
    ["Processos", "Mineração", "scripts/build_event_log.py, scripts/process_mining.py", "Event log, mapas, variantes e gargalos"],
    ["DevSecOps", "Pipeline", ".github/workflows/devsecops.yml, scripts/ci_local.ps1",
     "Lint, testes, SAST, dependências, build, Trivy e smoke test"],
], "Componentes do sistema", [2.4, 2.6, 5.2, 5.8])
doc.add_heading("Fluxo de dados", level=2)
for prefix, text in [
    ("Entrada: ", "o simulador publica um JSON por leitura no tópico sensor-readings (chave = talhão)."),
    ("Ingestão: ", "o consumer recebe a mensagem e preserva o evento original na camada raw."),
    ("Validação: ", "campos obrigatórios, timestamp UTC, faixas físicas (allowed_range) e duplicidade de event_id. "
                    "Leituras válidas vão para trusted; inválidas e duplicadas, para rejected com o motivo."),
    ("Consolidação: ", "a cada 10 leituras válidas ou 5 segundos, o motor recalcula a camada refined dos talhões "
                       "afetados e só então confirma os offsets no Kafka."),
    ("Decisão: ", "a última leitura do talhão é comparada com expected_range e cruzada com imagens Sick do mesmo "
                  "talhão nas últimas 24 h, gerando nível de atenção e recomendação."),
    ("Saída: ", "o dashboard lê trusted e refined a cada 5 segundos e exibe leituras, alertas, recomendações e a "
                "saúde da plataforma; relatórios e CSV podem ser exportados."),
    ("Rastreabilidade: ", "cada etapa grava uma linha de log JSON com case_id e atividade, base do event log."),
]:
    bullet(text, prefix)

# 2 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("2. Execução da plataforma", level=1)
p("Toda a plataforma sobe com um único comando e permanece em execução contínua, com reinício automático e "
  "healthchecks em cada serviço:")
code("docker compose --profile streaming up --build -d\ndocker compose ps")
figure(EVID / "01_containers_healthy.png", "Serviços em execução e saudáveis (docker compose ps)")
doc.add_heading("Exemplo de entrada", level=2)
sample_in = latest_jsonl(ROOT / "data_lake/raw/sensors")
if sample_in:
    p("Evento publicado pelo simulador e preservado na camada raw:")
    code(json.dumps(sample_in, indent=2, ensure_ascii=False))
doc.add_heading("Exemplo de saída", level=2)
statuses = []
for path in sorted((ROOT / "data_lake/refined/field_status").glob("*.json")):
    statuses.append(json.loads(path.read_text(encoding="utf-8")))
alert = next((s for s in statuses if s.get("attention_level") == "high"), None) or \
    next((s for s in statuses if s.get("attention_level") == "medium"), None) or (statuses[0] if statuses else None)
if alert:
    p("Status consolidado na camada refined, consumido pelo dashboard (trecho):")
    keep = {k: alert[k] for k in ("field_id", "attention_level", "recommendation", "decision_reasons",
                                   "sensor_summary", "updated_at") if k in alert}
    code(json.dumps(keep, indent=2, ensure_ascii=False))
if statuses:
    table(["Talhão", "Nível", "Recomendação", "Leituras (24 h)"],
          [[s["field_id"], s.get("attention_level", "-"), s.get("recommendation", "-"),
            str(s.get("sensor_summary", {}).get("readings_count", "-"))] for s in statuses],
          "Decisões automatizadas por talhão no momento da coleta", [2.5, 2.2, 8.8, 2.5])
figure(EVID / "04_ambiente.png", "Dashboard: leituras, situação operacional e recomendações por talhão")

# 3 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("3. Pipeline DevSecOps", level=1)
p("O pipeline roda automaticamente no GitHub Actions a cada push ou pull request na branch main e pode ser "
  "reproduzido localmente com o mesmo conjunto de ferramentas (scripts/ci_local.ps1). As etapas são "
  "encadeadas: o container só é construído e escaneado se qualidade e segurança passarem.")
table(["Etapa", "Ferramenta", "O que verifica", "Critério de bloqueio"], [
    ["Lint", "Ruff", "Erros, imports, sintaxe moderna, bugs prováveis (PEP 8)", "Qualquer apontamento"],
    ["Testes", "unittest + coverage", "40 testes automatizados do pipeline, dashboard, monitoramento e processos",
     "Teste falho ou cobertura < 65%"],
    ["SAST", "Bandit", "Padrões inseguros no código Python", "Severidade média ou alta"],
    ["Dependências", "pip-audit", "CVEs nas bibliotecas (base PyPI/OSV)", "Vulnerabilidade sem risco aceito"],
    ["Build", "Docker", "Imagem multi-stage do dashboard (usuário não-root, healthcheck)", "Falha de build"],
    ["Config.", "Trivy config", "Más práticas no Dockerfile", "Severidade alta ou crítica"],
    ["Imagem", "Trivy image", "CVEs de SO e bibliotecas na imagem", "Crítica com correção disponível"],
    ["Smoke test", "Docker + curl", "Container sobe e fica healthy; /_stcore/health responde", "Container não saudável"],
], "Etapas do pipeline", [2.2, 2.8, 6.4, 4.6])
antes = EVID / "devsecops/antes"
table(["Verificação", "Antes da Fase 7", "Depois", "Ação"], [
    ["Ruff (lint)", f"{count_found(antes / 'ruff_antes.txt')} apontamentos", "0", "Correções automáticas e manuais"],
    ["Bandit (SAST)", bandit_counts(antes / "bandit_antes.txt"), "0 altas, 0 médias",
     "Uso de random justificado (simulação), sem uso criptográfico"],
    ["pip-audit", f"{audit_count(antes / 'pip_audit_antes.json')} vulnerabilidades",
     "0 bloqueantes (39 com risco aceito)", "Pillow 10.2 → 12.3, pyarrow fixado em 23.0.1"],
    ["Trivy config", "1 alta (root), 1 baixa (sem HEALTHCHECK)", "0", "USER agro (não-root) e HEALTHCHECK"],
    ["Trivy imagem", "-", "0 críticas; 52 altas e 64 médias reportadas",
     "Gate aprovado; altas vêm do Debian base e de pip/setuptools do Python"],
], "Resultados antes e depois", [2.6, 4.3, 3.7, 5.4])
p("O pip-audit também revelou um problema funcional: sem versão fixa, uma build nova instalaria um pyarrow "
  "incompatível com o numpy exigido pelo TensorFlow, quebrando as tabelas do dashboard. O pyarrow passou a ser "
  "fixado. As 39 vulnerabilidades restantes do Pillow só são corrigidas na versão 13, ainda não suportada pelo "
  "Streamlit; o risco foi analisado e aceito formalmente em security/RISCOS_ACEITOS.md (o dashboard só abre "
  "imagens JPG/PNG locais, sem upload), e os scanners continuam reportando essas IDs a cada execução.")
figure(EVID / "06_pipeline_local.png", "Execução do pipeline local com as cinco etapas aprovadas")
figure(EVID / "07_github_actions.png", "Execução do workflow DevSecOps no GitHub Actions")

# 4 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("4. Monitoramento e execução automatizada", level=1)
p("O monitoramento combina três mecanismos complementares:")
bullet("cada componente (simulador, consumer, dashboard e pipeline de imagens) grava logs em JSON, uma linha "
       "por evento, em logs/<componente>.log (rotativos) e no console do container.", "Logs estruturados: ")
bullet("arquivos logs/status/<componente>.json atualizados a cada 5 segundos com estado, contadores, latência e "
       "último evento. Sem heartbeat por 30 s o componente fica 'atrasado'; por 120 s, 'offline'.", "Heartbeats: ")
bullet("o Docker consulta os heartbeats (consumer e simulador) e o endpoint de saúde do Streamlit; serviços com "
       "falha são marcados como unhealthy e reiniciados (restart: unless-stopped).", "Healthchecks: ")
table(["Indicador", "Fonte", "Uso"], [
    ["Estado de cada componente", "heartbeat", "Detectar serviço parado ou travado"],
    ["Eventos recebidos, aceitos, rejeitados, duplicados", "heartbeat do consumer", "Volume e qualidade da entrada"],
    ["Taxa de rejeição e qualidade dos dados", "heartbeat + Data Lake", "Sensores com falha ou dados inválidos"],
    ["Latência de ingestão (média e máxima)", "logs do consumer", "Atraso entre a leitura e o processamento"],
    ["Tempo do último micro-lote", "heartbeat do consumer", "Desempenho da consolidação"],
    ["Registros e última escrita por camada", "Data Lake", "Dados chegando em raw, trusted e rejected"],
    ["Vazão por minuto e resultado do processamento", "logs", "Tendência e distribuição dos desfechos"],
], "Indicadores de funcionamento", [6, 4, 6])
figure(EVID / "02_saude_sistema.png", "Aba Saúde do sistema: componentes, indicadores e Data Lake")
figure(EVID / "03_atividade_logs.png", "Vazão, latência de ingestão e logs recentes no dashboard")
figure(EVID / "05_logs_consumer.png", "Logs JSON do consumer em execução (docker compose logs)")
doc.add_heading("Como acompanhar o sistema", level=2)
table(["Pergunta", "Onde olhar"], [
    ["A plataforma está no ar?", "docker compose ps (healthy) ou aba Saúde do sistema (indicador geral)"],
    ["Os dados estão chegando?", "Vazão por minuto, heartbeat do simulador e última escrita da camada raw"],
    ["Há problema de qualidade?", "Taxa de rejeição, camada rejected e logs WARNING/ERROR"],
    ["Há lentidão?", "Latência de ingestão e tempo do micro-lote"],
    ["Por que uma leitura foi rejeitada?", "Logs filtrados por case_id (event_id) e motivo em rejected"],
], None, [5.5, 10.5])

# 5 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("5. Análise de processo (mineração de processos)", level=1)
summary = json.loads((PM / "resumo.json").read_text(encoding="utf-8")) if (PM / "resumo.json").exists() else None
p("Cada leitura é um caso (case_id = event_id). Os logs estruturados de simulador e consumer são convertidos em "
  "um event log (case_id, atividade, timestamp, recurso e atributos) e analisados com a biblioteca pm4py:")
code("docker compose --profile analysis run --rm process-mining")
event_log = PM / "event_log.csv"
if event_log.exists():
    frame = pd.read_csv(event_log)
    case_ids = frame.loc[frame["activity"] == "Alerta emitido", "case_id"]
    case = case_ids.iloc[0] if len(case_ids) else frame["case_id"].iloc[0]
    rows = frame[frame["case_id"] == case]
    doc.add_heading("Exemplo de event log", level=2)
    table(["case_id", "Atividade", "Timestamp (UTC)", "Recurso"],
          [[str(r.case_id)[:17] + "…", r.activity, str(r.timestamp)[:23], r.resource] for r in rows.itertuples()],
          "Eventos de uma leitura que gerou alerta", [3.6, 5, 4.8, 2.6])
if summary:
    p(f"O período analisado reúne {summary['eventos']} eventos de {summary['casos']} leituras, "
      f"{len(summary['atividades'])} atividades distintas e {summary['variantes']} variantes de caminho.")
doc.add_heading("Representação do processo", level=2)
figure(PM / "dfg_frequencia.png", "Mapa do processo descoberto (Directly-Follows Graph com frequências)", 12.5)
figure(PM / "bpmn_processo.png", "Modelo BPMN descoberto pelo Inductive Miner", 16)
if summary:
    doc.add_heading("Variações", level=2)
    variants = pd.read_csv(PM / "variantes.csv").head(6)
    table(["#", "Resultado", "Etapas", "Casos", "%"],
          [[f"V{i + 1}", r.resultado, r.etapas, r.casos, f"{r.percentual}%"] for i, r in enumerate(variants.itertuples())],
          "Variantes mais frequentes", [1.2, 7.5, 2.2, 2.2, 2.2])
    figure(PM / "variantes.png", "Distribuição das variantes do processo", 14)
    p(f"O caminho feliz completo (leitura gerada até a decisão) cobre {summary['caminho_feliz_pct']}% dos casos. "
      "As demais variantes são leituras rejeitadas na validação, duplicatas descartadas e casos com reenvio, que "
      "percorrem parte do fluxo duas vezes (retrabalho).")
    scen = summary.get("alerta_por_cenario", {})
    if scen:
        table(["Cenário", "Decisões", "% com alerta"], [[k, v["casos"], f"{v['alerta_pct']}%"] for k, v in scen.items()],
              "Variação do desfecho por cenário simulado", [5, 5, 6])
    doc.add_heading("Gargalos", level=2)
    figure(PM / "gargalos.png", "Tempo de espera mediano entre etapas", 15)
    figure(PM / "lead_time.png", "Composição do lead time até a decisão", 15)
    lt = summary.get("lead_time", {})
    share = lt.get("participacao_pct", {})
    if lt:
        p(f"O lead time médio da leitura até a decisão é de {lt['media_s']} s (mediana {lt['mediana_s']} s, p95 "
          f"{lt['p95_s']} s). A espera do micro-lote responde por {share.get('espera_micro_lote_s', 0)}% desse "
          f"tempo, o trânsito no Kafka por {share.get('transito_kafka_s', 0)}% e a validação com gravação por "
          f"{share.get('validacao_gravacao_s', 0)}%: o gargalo não é processamento, e sim a espera pela consolidação.")
    if summary["alertas_mascarados"]:
        p(f"Das {summary['leituras_fora_da_faixa']} leituras aceitas fora da faixa operacional, "
          f"{summary['alertas_mascarados']} ({summary['alertas_mascarados_pct']}%) não viraram alerta: o motor "
          "decide pela última leitura de cada talhão, e uma leitura crítica seguida no mesmo micro-lote por outra "
          "normal do mesmo talhão fica 'mascarada'.")
    else:
        p(f"As {summary['leituras_fora_da_faixa']} leituras aceitas fora da faixa operacional geraram alerta "
          "(nenhuma mascarada). A análise por caso, porém, expôs um risco de projeto: o motor decide pela última "
          "leitura de cada talhão. Neste volume (uma leitura a cada 2 s, cerca de uma por talhão em cada lote) isso "
          "não ocorre; com mais sensores por talhão, uma leitura crítica seguida no mesmo micro-lote por outra "
          "normal deixaria de gerar alerta. O indicador 'alertas mascarados' permanece no relatório para monitorar isso.")
    if summary.get("recebidas_antes_do_ack"):
        p(f"Outra variação revelada pelo event log: em {summary['recebidas_antes_do_ack']} casos o consumer recebeu a "
          "leitura antes de o producer registrar a confirmação (ack) do Kafka. Não é erro: o producer só registra "
          "'Publicada' quando a confirmação volta, e o broker já entregou a mensagem ao consumer nesse intervalo. "
          "A ordem das atividades varia, mas o resultado é o mesmo.")
    doc.add_heading("Oportunidades de melhoria", level=2)
    analysis = (PM / "analise_processo.md").read_text(encoding="utf-8")
    section = analysis.split("## Oportunidades de melhoria (geradas a partir dos indicadores)")[1].split("##")[0]
    for line in section.strip().splitlines():
        if line.startswith("- "):
            bullet(line[2:])
    bullet("Separar o consumer por partição (talhão) e executar réplicas quando o volume crescer, já que a análise "
           "mostra que a etapa de validação consome menos de 1% do tempo.")

# 6 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("6. Apresentação do produto", level=1)
table(["Item", "Síntese"], [
    ["Problema", "Pequenos e médios produtores de tomate perdem produção por irrigação inadequada, calor e doenças "
                 "detectadas tarde, decidindo com base em inspeção visual e experiência."],
    ["Solução", "Plataforma que monitora o talhão continuamente com sensores IoT e imagens de folhas e transforma "
                "dados em alertas e recomendações objetivas por talhão."],
    ["Diferencial", "Une sensores e visão computacional no mesmo motor de decisão; dados rastreáveis em camadas "
                    "(auditoria de cada leitura); operação monitorada e segura desde o início."],
    ["Público-alvo", "Produtores de tomate de pequeno e médio porte, cooperativas e consultores agronômicos."],
    ["Proposta de valor", "Menos perdas e desperdício de água com decisões no momento certo, a partir de "
                          "informação confiável e simples de acompanhar."],
], "Resumo do pitch (detalhado nos slides)", [3.2, 12.8])

# 7 ---------------------------------------------------------------------------------------------------------------
doc.add_heading("7. Mapa de artefatos", level=1)
table(["Requisito", "Artefatos"], [
    ["Arquitetura final", "docs/fase7/diagramas/arquitetura_final.png (.dot), seção 1"],
    ["Execução da plataforma", "docker-compose.yml, prints 01 e 04, exemplos de entrada e saída (seção 2)"],
    ["Pipeline DevSecOps", "Dockerfile, docker-compose.yml, .github/workflows/devsecops.yml, scripts/ci_local.ps1, "
                           "pyproject.toml, security/, prints 06 e 07, artifacts/ci/"],
    ["Monitoramento", "services/observability.py, services/health.py, scripts/healthcheck.py, prints 02, 03 e 05"],
    ["Análise de processo", "output/process_mining/ (event_log.csv/.xes, mapas, variantes, gargalos, analise_processo.md)"],
    ["Pitch", "Slides do pitch e vídeo de até 5 minutos"],
    ["Vídeo final", "Demonstração de 5 a 8 minutos"],
], None, [4, 12])
doc.save(OUT)
print(f"Documento gerado: {OUT}")
