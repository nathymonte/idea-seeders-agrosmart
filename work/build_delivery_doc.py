from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "Artefatos_Data_Lake_AgroSmart.docx"
ARCH_IMG = DOCS / "diagrama_arquitetura_data_lake.png"
INGEST_IMG = DOCS / "diagrama_fluxo_ingestao.png"
BUSINESS_IMG = DOCS / "diagrama_modelo_negocio_dados.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#355C7D") -> None:
    draw.line([start, end], fill=color, width=4)
    ex, ey = end
    sx, sy = start
    if ex >= sx:
        points = [(ex, ey), (ex - 14, ey - 8), (ex - 14, ey + 8)]
    else:
        points = [(ex, ey), (ex + 14, ey - 8), (ex + 14, ey + 8)]
    draw.polygon(points, fill=color)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    fill: str,
    outline: str = "#38516B",
) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = xy
    draw.text((x1 + 18, y1 + 14), title, fill="#16324F", font=font(25, True))
    y = y1 + 52
    for line in lines:
        draw.text((x1 + 18, y), line, fill="#203040", font=font(18))
        y += 28


def create_architecture_diagram() -> None:
    img = Image.new("RGB", (1600, 900), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((50, 32), "Arquitetura de Dados - Data Lake AgroSmart", fill="#16324F", font=font(38, True))

    rounded_box(draw, (70, 145, 360, 350), "Fontes", [
        "Sensores IoT simulados",
        "CSV historico",
        "Imagens das plantas",
    ], "#EAF4EA")

    rounded_box(draw, (470, 90, 760, 250), "Ingestao", [
        "Kafka producer/consumer",
        "historical_ingestion.py",
        "upload/classificacao",
    ], "#EEF3FA")

    rounded_box(draw, (470, 310, 760, 470), "Raw", [
        "Dado original preservado",
        "JSONL particionado",
        "CSV copiado sem alteracao",
    ], "#FFF5DF")

    rounded_box(draw, (850, 90, 1140, 250), "Trusted", [
        "Validado e tipado",
        "Sem duplicidade",
        "Contrato padronizado",
    ], "#E7F4F9")

    rounded_box(draw, (850, 310, 1140, 470), "Rejected", [
        "Eventos invalidos",
        "Motivo da rejeicao",
        "Auditoria de qualidade",
    ], "#FCECEC")

    rounded_box(draw, (1230, 185, 1520, 385), "Refined", [
        "Medias por talhao",
        "Nivel de atencao",
        "Recomendacao",
        "field_status/*.json",
    ], "#ECEAF7")

    rounded_box(draw, (1010, 600, 1330, 800), "Analise", [
        "Dashboard Streamlit",
        "Relatorio agricola",
        "Apoio a decisao",
    ], "#F2F6F7")

    arrow(draw, (360, 245), (470, 170))
    arrow(draw, (610, 250), (610, 310))
    arrow(draw, (760, 390), (850, 170))
    arrow(draw, (760, 390), (850, 390))
    arrow(draw, (1140, 170), (1230, 280))
    arrow(draw, (1375, 385), (1200, 600))
    arrow(draw, (1010, 700), (760, 470))

    img.save(ARCH_IMG)


def create_ingestion_diagram() -> None:
    img = Image.new("RGB", (1600, 900), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((50, 32), "Fluxo de Ingestao, Validacao e Manutencao", fill="#16324F", font=font(38, True))

    boxes = [
        ((70, 160, 350, 320), "Entrada", ["Kafka sensor-readings", "CSV historico", "Imagens processadas"], "#EAF4EA"),
        ((445, 160, 725, 320), "Raw", ["Salva evento original", "Preserva arquivo CSV", "Particiona por data"], "#FFF5DF"),
        ((820, 80, 1100, 240), "Validador", ["Campos obrigatorios", "Faixas numericas", "Timestamp", "Duplicidade"], "#EEF3FA"),
        ((820, 320, 1100, 480), "Rejected", ["Evento invalido", "Erros detalhados", "Rastreabilidade"], "#FCECEC"),
        ((1195, 80, 1475, 240), "Trusted", ["Evento normalizado", "schema_version 1.0", "Pronto para analise"], "#E7F4F9"),
        ((1195, 320, 1475, 500), "Refined", ["Resumo por talhao", "Nivel de atencao", "Recomendacao"], "#ECEAF7"),
        ((615, 620, 985, 790), "Manutencao", ["Separar dado gerado do codigo", "Reprocessar refined", "Monitorar rejected", "Versionar amostras pequenas"], "#F2F6F7"),
    ]
    for xy, title, lines, fill in boxes:
        rounded_box(draw, xy, title, lines, fill)

    arrow(draw, (350, 240), (445, 240))
    arrow(draw, (725, 240), (820, 160))
    arrow(draw, (960, 240), (960, 320))
    arrow(draw, (1100, 160), (1195, 160))
    arrow(draw, (1335, 240), (1335, 320))
    arrow(draw, (960, 480), (760, 620))
    arrow(draw, (1335, 500), (905, 620))

    img.save(INGEST_IMG)


def create_business_diagram() -> None:
    img = Image.new("RGB", (1600, 900), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((50, 32), "Modelo de Negocio Orientado a Dados - AgroSmart", fill="#16324F", font=font(38, True))

    rounded_box(draw, (90, 170, 395, 360), "Dados", [
        "Sensores do campo",
        "Imagens das plantas",
        "Historico por talhao",
    ], "#EAF4EA")

    rounded_box(draw, (500, 120, 820, 310), "Inteligencia", [
        "Data Lake",
        "Indicadores refinados",
        "IA de recomendacao",
    ], "#EEF3FA")

    rounded_box(draw, (925, 170, 1245, 360), "Valor", [
        "Alertas preventivos",
        "Decisao mais rapida",
        "Menos perda no cultivo",
    ], "#ECEAF7")

    rounded_box(draw, (520, 520, 830, 720), "Receita", [
        "Assinatura SaaS",
        "Plano por talhao",
        "Relatorios premium",
        "Implantacao assistida",
    ], "#FFF5DF")

    rounded_box(draw, (950, 520, 1285, 720), "Parcerias", [
        "Cooperativas",
        "Agronomos",
        "Fornecedores IoT",
        "Instituicoes de ensino",
    ], "#F2F6F7")

    arrow(draw, (395, 265), (500, 215))
    arrow(draw, (820, 215), (925, 265))
    arrow(draw, (1085, 360), (700, 520))
    arrow(draw, (1085, 360), (1085, 520))
    arrow(draw, (830, 620), (950, 620))

    img.save(BUSINESS_IMG)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Calibri"
    run.font.size = Pt(10)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for index, header in enumerate(headers):
        set_cell_text(hdr[index], header, True)
        set_cell_shading(hdr[index], "F2F4F7")
        hdr[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)


def add_code_block(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(32, 48, 64)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "F4F6F9")
    paragraph._p.get_or_add_pPr().append(shading)


def add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)


def build_doc() -> None:
    DOCS.mkdir(exist_ok=True)
    create_architecture_diagram()
    create_ingestion_diagram()
    create_business_diagram()

    field_status = json.loads((ROOT / "data_lake/refined/field_status/TALHAO-01.json").read_text(encoding="utf-8"))
    sample_csv = (ROOT / "samples/sensor_history.csv").read_text(encoding="utf-8").strip()
    sample_json = (ROOT / "samples/valid_sensor_event.json").read_text(encoding="utf-8").strip()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(11)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        styles[name].font.name = "Calibri"
        styles[name].font.color.rgb = RGBColor(46, 116, 181)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)
    run = title.add_run("Artefatos de Entrega - Data Lake AgroSmart")
    run.bold = True
    run.font.name = "Calibri"
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(22, 50, 79)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Arquitetura de dados, ingestao, manutencao e IA generativa simulada para apoio a decisao")
    subtitle.paragraph_format.space_after = Pt(18)

    doc.add_heading("1. Avaliacao dos requisitos", level=1)
    doc.add_paragraph(
        "A implementacao atual atende aos requisitos principais da sprint. O projeto possui fontes estruturadas "
        "e nao estruturadas, camadas de Data Lake, ingestao por CSV e Kafka, validacao de confiabilidade, "
        "geracao de indicadores refinados e uma saida de recomendacao para apoiar o produtor rural."
    )
    add_table(
        doc,
        ["Requisito", "Status", "Evidencia no projeto"],
        [
            [
                "Arquitetura de Data Lake",
                "Cumprido",
                "Camadas data_lake/raw, trusted, refined e rejected; servico services/data_lake.py; dashboard lendo refined.",
            ],
            [
                "Fontes estruturadas e nao estruturadas",
                "Cumprido",
                "CSV historico e JSON de sensores; imagens de folhas processadas pelo modelo de classificacao.",
            ],
            [
                "Ingestao e confiabilidade",
                "Cumprido",
                "historical_ingestion.py, streaming consumer/producer, validacao de campos, faixas, timestamps e duplicidade.",
            ],
            [
                "IA generativa para apoio a decisao",
                "Cumprido como simulacao",
                "Relatorio agricola e recomendacao textual gerada a partir dos dados refinados e das classificacoes.",
            ],
        ],
        [2.0, 1.4, 3.1],
    )

    doc.add_heading("2. Arquitetura de Dados com Data Lake", level=1)
    doc.add_paragraph(
        "A arquitetura proposta organiza os dados do AgroSmart desde a captura ate a analise. As fontes incluem "
        "sensores IoT simulados, arquivo CSV historico e imagens de plantas classificadas por visao computacional."
    )
    doc.add_picture(str(ARCH_IMG), width=Inches(6.5))
    doc.add_paragraph("Diagrama 1 - Arquitetura do Data Lake AgroSmart.")

    doc.add_heading("Funcao de cada camada", level=2)
    add_bullet(doc, "Raw: armazena os dados originais, sem transformacao, para preservar rastreabilidade e permitir reprocessamentos.")
    add_bullet(doc, "Trusted: armazena dados validados, tipados, padronizados e sem duplicidade, prontos para uso analitico.")
    add_bullet(doc, "Rejected: guarda eventos rejeitados e os motivos de rejeicao, apoiando auditoria e melhoria da qualidade dos dados.")
    add_bullet(doc, "Refined: consolida informacoes por talhao, calculando indicadores, nivel de atencao e recomendacao operacional.")

    doc.add_heading("3. Processo de Ingestao, Administracao e Manutencao", level=1)
    doc.add_paragraph(
        "O fluxo de ingestao aceita dados em lote e em streaming. O CSV historico passa por preservacao na camada raw "
        "e validacao linha a linha. Os eventos Kafka sao recebidos pelo consumer, gravados em raw, validados e enviados "
        "para trusted ou rejected. Sempre que dados confiaveis sao atualizados, a camada refined pode ser recalculada."
    )
    doc.add_picture(str(INGEST_IMG), width=Inches(6.5))
    doc.add_paragraph("Diagrama 2 - Fluxo de ingestao e manutencao.")

    doc.add_heading("Exemplo de dados ingeridos", level=2)
    doc.add_paragraph("Exemplo CSV historico:")
    add_code_block(doc, sample_csv)
    doc.add_paragraph("Exemplo JSON de sensor:")
    add_code_block(doc, sample_json)

    doc.add_heading("Organizacao e manutencao", level=2)
    add_bullet(doc, "Os dados sao particionados por ano, mes e dia, facilitando consulta, auditoria e descarte futuro.")
    add_bullet(doc, "Arquivos gerados pelo Data Lake nao precisam ser versionados no Git; o codigo recria pastas e artefatos.")
    add_bullet(doc, "A camada rejected deve ser monitorada para identificar falhas recorrentes de origem ou contrato.")
    add_bullet(doc, "A camada refined pode ser regenerada a partir da trusted, mantendo o dashboard consistente.")
    add_bullet(doc, "Amostras pequenas em samples/ devem ser versionadas para demonstracao e testes.")

    doc.add_heading("4. Inteligencia Artificial Generativa para Apoio a Decisao", level=1)
    doc.add_paragraph(
        "Nesta sprint, a IA generativa foi representada por uma saida textual simulada, baseada no contexto calculado "
        "pelo Data Lake e no resultado do modelo de classificacao de imagens. A recomendacao nao substitui a validacao "
        "humana; ela organiza os sinais relevantes para acelerar a tomada de decisao."
    )

    doc.add_heading("Contexto utilizado", level=2)
    sensor = field_status["sensor_summary"]
    add_table(
        doc,
        ["Campo", "Valor"],
        [
            ["Talhao", field_status["field_id"]],
            ["Cultura", field_status["crop"]],
            ["Nivel de atencao", field_status["attention_level"]],
            ["Umidade media do solo", f"{sensor['average_soil_moisture_percent']}%"],
            ["Temperatura media", f"{sensor['average_air_temperature_celsius']} C"],
            ["Umidade media do ar", f"{sensor['average_air_humidity_percent']}%"],
            ["Imagens Sick", str(field_status["latest_image_analysis"]["sick_images"])],
        ],
        [2.3, 4.2],
    )

    doc.add_heading("Exemplo de saida da IA", level=2)
    ai_text = (
        "Relatorio AgroSmart - Talhao TALHAO-01\n\n"
        "O talhao apresenta nivel de atencao medio. As medias ambientais estao proximas da faixa operacional, "
        "mas as classificacoes de imagem indicam ocorrencias de folhas com possivel problema fitossanitario. "
        "Recomenda-se realizar inspeção visual no talhao, priorizando as plantas associadas as imagens classificadas "
        "como Sick. Caso a condicao seja confirmada em campo, o produtor deve intensificar o monitoramento nas proximas "
        "24 horas e registrar novas imagens para comparar a evolucao."
    )
    add_code_block(doc, ai_text)
    doc.add_paragraph(
        "Funcionamento: o sistema le os dados refinados do Data Lake, resume os indicadores principais e transforma "
        "esses sinais em uma recomendacao textual objetiva. Em uma versao futura, essa etapa poderia chamar uma API "
        "de IA generativa; para a sprint, a simulacao documenta o comportamento esperado."
    )

    doc.add_heading("5. Modelo de Negocio e Inteligencia de Dados", level=1)
    doc.add_paragraph(
        "O modelo de negocio do AgroSmart deve evoluir de um prototipo de classificacao de folhas para uma solucao "
        "orientada a dados. O valor principal passa a estar na combinacao entre monitoramento continuo, historico "
        "do cultivo, classificacao de imagens e recomendacoes objetivas para o produtor."
    )
    doc.add_picture(str(BUSINESS_IMG), width=Inches(6.5))
    doc.add_paragraph("Diagrama 3 - Modelo de negocio orientado a dados.")

    doc.add_heading("Proposta de valor", level=2)
    add_bullet(doc, "Ajudar produtores de tomate a identificar sinais de risco antes que o problema gere perda relevante na lavoura.")
    add_bullet(doc, "Centralizar sensores, imagens e historico em um unico painel de acompanhamento por talhao.")
    add_bullet(doc, "Transformar dados tecnicos em recomendacoes simples, rastreaveis e acionaveis.")
    add_bullet(doc, "Reduzir dependencia de acompanhamento manual constante, mantendo a decisao final com o produtor ou responsavel tecnico.")

    doc.add_heading("Fontes de receita", level=2)
    add_table(
        doc,
        ["Fonte", "Como funcionaria", "Valor para o cliente"],
        [
            [
                "Assinatura SaaS",
                "Planos mensais por produtor, propriedade ou quantidade de talhoes monitorados.",
                "Acesso continuo ao dashboard, historico e recomendacoes.",
            ],
            [
                "Plano por volume de dados",
                "Cobranca por quantidade de sensores, imagens analisadas ou eventos ingeridos.",
                "Permite iniciar pequeno e crescer conforme a operacao rural aumenta.",
            ],
            [
                "Relatorios premium",
                "Relatorios periodicos com tendencias, comparativos e recomendacoes consolidadas.",
                "Apoia reunioes tecnicas, prestacao de contas e tomada de decisao.",
            ],
            [
                "Implantacao e suporte",
                "Servicos de configuracao, treinamento e adaptacao dos limiares por cultura.",
                "Reduz barreiras de adocao para produtores e cooperativas.",
            ],
        ],
        [1.45, 2.65, 2.4],
    )

    doc.add_heading("Monetizacao baseada em dados", level=2)
    doc.add_paragraph(
        "A monetizacao baseada em dados deve respeitar privacidade, consentimento e agregacao. O AgroSmart pode usar "
        "dados historicos e indicadores consolidados para oferecer inteligencia adicional sem expor informacoes "
        "sensivelmente identificaveis de uma propriedade individual."
    )
    add_bullet(doc, "Benchmarks anonimizados por regiao, cultura ou periodo, comparando indicadores medios de saude das plantas.")
    add_bullet(doc, "Alertas inteligentes baseados em padroes historicos, como queda recorrente de umidade ou aumento de ocorrencias Sick.")
    add_bullet(doc, "Relatorios comparativos de safra, permitindo acompanhar evolucao de risco, produtividade esperada e recorrencia de anomalias.")
    add_bullet(doc, "API ou exportacao de dados refinados para sistemas de gestao agricola, cooperativas ou consultorias agronomicas.")

    doc.add_heading("Parcerias estrategicas", level=2)
    add_table(
        doc,
        ["Parceiro", "Papel na solucao", "Beneficio estrategico"],
        [
            [
                "Cooperativas agricolas",
                "Distribuicao da solucao e apoio aos produtores associados.",
                "Acesso a base de usuarios e validacao em campo.",
            ],
            [
                "Agronomos e consultorias",
                "Interpretacao tecnica dos alertas e ajuste de recomendacoes.",
                "Aumenta confiabilidade e aderencia agronomica.",
            ],
            [
                "Fornecedores de IoT",
                "Sensores fisicos, conectividade e manutencao de dispositivos.",
                "Permite evoluir de simulacao para monitoramento real.",
            ],
            [
                "Instituicoes de ensino e pesquisa",
                "Aprimoramento dos modelos, validacao de dados e experimentos.",
                "Fortalece a qualidade cientifica e a inovacao do produto.",
            ],
            [
                "Empresas de insumos",
                "Ofertas integradas de manejo, desde que separadas da recomendacao tecnica.",
                "Cria canais comerciais sem comprometer a imparcialidade do apoio a decisao.",
            ],
        ],
        [1.55, 2.45, 2.5],
    )

    doc.add_heading("Uso da inteligencia de dados no negocio", level=2)
    doc.add_paragraph(
        "A inteligencia de dados sustenta tanto o produto quanto a monetizacao. No produto, ela alimenta alertas, "
        "dashboards e recomendacoes. No negocio, ela permite segmentar planos, demonstrar valor ao cliente, criar "
        "relatorios premium e desenvolver parcerias baseadas em evidencias. O diferencial do AgroSmart nao e apenas "
        "coletar dados, mas transformar leituras, imagens e historico em decisao pratica para o campo."
    )

    doc.add_heading("6. Artefatos entregues neste documento", level=1)
    add_bullet(doc, "Diagrama da arquitetura do Data Lake.")
    add_bullet(doc, "Explicacao textual da funcao de cada camada.")
    add_bullet(doc, "Diagrama do fluxo de ingestao.")
    add_bullet(doc, "Exemplos de dados ingeridos em CSV e JSON.")
    add_bullet(doc, "Documento curto de organizacao e manutencao dos dados.")
    add_bullet(doc, "Exemplo de saida de IA generativa simulada e descricao do contexto utilizado.")
    add_bullet(doc, "Modelo de negocio orientado a dados com proposta de valor, receitas, monetizacao e parcerias.")

    doc.save(OUT)


if __name__ == "__main__":
    build_doc()
