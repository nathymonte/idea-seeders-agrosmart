import pandas as pd
from pathlib import Path
from datetime import datetime

CSV_PATH = "output/classificacoes.csv"
OUTPUT_DIR = "output"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Arquivo carregado com sucesso: {CSV_PATH}")

except FileNotFoundError:
    print(f"[ERRO] Arquivo não encontrado: {CSV_PATH}")
    exit()

total_imagens = len(df)

folhas_doentes = len(
    df[df["predicted_label"] == "Sick"]
)

folhas_saudaveis = len(
    df[df["predicted_label"] == "Health"]
)

percentual_doentes = (
    folhas_doentes / total_imagens
) * 100

confidence_media = df[df["is_correct"] == True]["confidence"].mean()

anomalias_criticas = df[
    (df["anomalia"] == "Ferrugem")
    | (df["anomalia"] == "Praga mastigadora")
]

prioridade_alta = len(anomalias_criticas) > 0

necessita_monitoramento = not (total_imagens == folhas_saudaveis)

talhoes_afetados = (
    df[df["predicted_label"] == "Sick"]["localidade"]
    .value_counts()
)

talhao_critico = (
    talhoes_afetados.idxmax()
    if not talhoes_afetados.empty
    else "Nenhum"
)

quantidade_talhao_critico = (
    talhoes_afetados.max()
    if not talhoes_afetados.empty
    else 0
)

percentual_talhao_critico = (
    (
        quantidade_talhao_critico
        / folhas_doentes
    ) * 100
    if folhas_doentes > 0
    else 0
)

anomalias_detectadas = (
    df[df["anomalia"] != "Nenhuma"]["anomalia"]
    .value_counts()
)

principal_anomalia = (
    anomalias_detectadas.idxmax()
    if not anomalias_detectadas.empty
    else "Nenhuma"
)

lista_anomalias = []

for anomalia, quantidade in anomalias_detectadas.items():

    lista_anomalias.append(
        f"- {anomalia}: {quantidade} ocorrência(s)"
    )

if not lista_anomalias:

    lista_anomalias.append(
        "- Nenhuma anomalia relevante identificada"
    )

recomendacoes = []

if talhao_critico != "Nenhum":

    recomendacoes.append(
        f"- [ALTA PRIORIDADE] Realizar inspeção presencial no {talhao_critico}."
    )

if prioridade_alta:

    recomendacoes.append(
        "- [AÇÃO IMEDIATA] Avaliar aplicação preventiva de controle fitossanitário."
    )

if necessita_monitoramento:

    recomendacoes.append(
        "- Intensificar monitoramento da plantação nas próximas 24 horas."
    )

if folhas_doentes > 0:

    recomendacoes.append(
        "- Realizar nova coleta de imagens para acompanhamento da evolução das anomalias."
    )

if folhas_doentes == 0:

    recomendacoes.append(
        "- Manter rotina atual de monitoramento preventivo da plantação."
    )

if percentual_doentes >= 50:

    status_final = "CRÍTICO"
    nivel_risco = "ALTO"
    severidade = "ALTA"

elif percentual_doentes >= 30:

    status_final = "ALERTA"
    nivel_risco = "MODERADO"
    severidade = "MODERADA"

else:

    status_final = "CONTROLADO"
    nivel_risco = "BAIXO"
    severidade = "BAIXA"

if status_final == "CRÍTICO":

    resumo_executivo = f"""
O sistema identificou alta incidência de folhas com indícios de doenças agrícolas no conjunto analisado.

Foram detectadas anomalias compatíveis com ferrugem, mancha foliar e pragas mastigadoras.

O {talhao_critico} apresentou o maior número de ocorrências e deve ser tratado como prioridade operacional.
"""

elif status_final == "ALERTA":

    resumo_executivo = f"""
O sistema identificou sinais moderados de anomalias agrícolas no conjunto analisado.

Embora o cenário ainda não seja considerado crítico, o {talhao_critico} apresentou maior concentração de ocorrências e requer atenção preventiva.
"""

else:

    resumo_executivo = """
O sistema não identificou riscos agrícolas relevantes no conjunto analisado.

Os indicadores atuais sugerem estabilidade da plantação monitorada.
"""

meses = {
    1: "janeiro",
    2: "fevereiro",
    3: "março",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro"
}

agora = datetime.now()
data_formatada = f"{agora.day} de {meses[agora.month]} de {agora.year} às {agora.strftime('%H:%M')}"
timestamp = agora.strftime("%Y%m%d_%H%M")

nome_relatorio = (
    f"relatorio_agricola_{timestamp}.md"
)

caminho_relatorio = (
    Path(OUTPUT_DIR) / nome_relatorio
)

relatorio = f"""
# RELATÓRIO DE MONITORAMENTO AGRÍCOLA

Data de geração: {data_formatada}

---

# RESUMO

- Status operacional: {status_final}
- Nível de risco agrícola: {nivel_risco}
- Severidade estimada: {severidade}
- Talhão prioritário: {talhao_critico}
- Principal anomalia detectada: {principal_anomalia}
- Confiabilidade média da análise: {confidence_media:.2f}%

---

# ANÁLISE GERAL

{resumo_executivo}

---

# INDICADORES DA ANÁLISE

- Total de imagens analisadas: {total_imagens}
- Folhas saudáveis identificadas: {folhas_saudaveis}
- Folhas com indícios de doença: {folhas_doentes}
- Percentual de incidência identificado: {percentual_doentes:.2f}%

---

# LOCALIDADES MAIS AFETADAS

- Talhão prioritário: {talhao_critico}
- Ocorrências registradas: {quantidade_talhao_critico}
- Participação nas ocorrências totais: {percentual_talhao_critico:.2f}%

---

# PRINCIPAIS ANOMALIAS IDENTIFICADAS

{chr(10).join(lista_anomalias)}

---

# RECOMENDAÇÕES

{chr(10).join(recomendacoes)}

---

# OBSERVAÇÃO FINAL

Este relatório foi gerado pelo sistema AgroSmart com base na análise automatizada das imagens agrícolas processadas pelo modelo de visão computacional.

O objetivo da automação é auxiliar produtores rurais na identificação preventiva de possíveis riscos agrícolas e apoiar tomadas de decisão mais rápidas no monitoramento da plantação.

---
"""

with open(
    caminho_relatorio,
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(relatorio)

print("\n====================================")
print("RELATÓRIO AGRÍCOLA GERADO")
print("====================================")
print(f"Arquivo: {caminho_relatorio}")
print(f"Status operacional: {status_final}")
print(f"Nível de risco: {nivel_risco}")
print("====================================\n")