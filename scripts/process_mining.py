"""Mineracao de processos do AgroSmart a partir do event log.

Gera, em output/process_mining/:
- dfg_frequencia.png / dfg_performance.png: mapa do processo (Directly-Follows Graph) com pm4py;
- bpmn_processo.png: modelo descoberto pelo Inductive Miner;
- variantes.csv / variantes.png: caminhos distintos percorridos pelas leituras;
- transicoes.csv / gargalos.png: tempo de espera entre etapas consecutivas;
- lead_time.png: tempo total da leitura gerada ate a decisao;
- resumo.json e analise_processo.md: indicadores e insights (gargalos, variacoes, melhorias).

Uso:
    python -m scripts.process_mining --event-log output/process_mining/event_log.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.paths import PROJECT_ROOT  # noqa: E402

HAPPY_PATH = ("Leitura gerada", "Publicada no Kafka", "Recebida pelo consumer", "Gravada na raw", "Validada",
              "Gravada na trusted", "Consolidada na refined", "Decisao gerada")
OUTCOMES = {"Alerta emitido": "Aceita com alerta", "Decisao gerada": "Aceita sem alerta",
            "Rejeitada (validacao)": "Rejeitada na validacao", "Rejeitada (payload invalido)": "Payload invalido",
            "Descartada (duplicada)": "Duplicata descartada"}
INK, MUTED, GRID, ACCENT = "#172b25", "#5c6f68", "#e3e8e6", "#187b57"


def load_event_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame.sort_values(["case_id", "timestamp"], kind="stable").reset_index(drop=True)


def variants(frame: pd.DataFrame) -> pd.DataFrame:
    traces = frame.groupby("case_id", sort=False)["activity"].agg(tuple)
    table = traces.value_counts().rename_axis("variante").reset_index(name="casos")
    table["percentual"] = (table["casos"] / table["casos"].sum() * 100).round(1)
    table["etapas"] = table["variante"].map(len)
    table["resultado"] = table["variante"].map(classify_trace)
    table["caminho"] = table["variante"].map(" -> ".join)
    return table.drop(columns="variante")


def classify_trace(trace: tuple[str, ...]) -> str:
    label = next((OUTCOMES[a] for a in OUTCOMES if a in trace), "Incompleto")
    if trace.count("Recebida pelo consumer") > 1:
        label += " + reenvio"
    return label


def transitions(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["proxima"] = ordered.groupby("case_id")["activity"].shift(-1)
    ordered["espera_s"] = (ordered.groupby("case_id")["timestamp"].shift(-1) - ordered["timestamp"]).dt.total_seconds()
    pairs = ordered.dropna(subset=["proxima"])
    table = pairs.groupby(["activity", "proxima"])["espera_s"].agg(
        ocorrencias="count", media_s="mean", mediana_s="median", p95_s=lambda s: s.quantile(0.95), max_s="max"
    ).reset_index().rename(columns={"activity": "de", "proxima": "para"})
    table["tempo_total_s"] = table["ocorrencias"] * table["media_s"]
    return table.sort_values("tempo_total_s", ascending=False).round(4)


def lead_times(frame: pd.DataFrame) -> pd.DataFrame:
    """Tempo por fase para leituras que chegaram a decisao."""
    pivot = frame.pivot_table(index="case_id", columns="activity", values="timestamp", aggfunc="min")
    needed = ["Leitura gerada", "Publicada no Kafka", "Recebida pelo consumer", "Gravada na trusted",
              "Consolidada na refined", "Decisao gerada"]
    pivot = pivot.dropna(subset=[c for c in needed if c in pivot])
    if pivot.empty or not set(needed) <= set(pivot.columns):
        return pd.DataFrame()
    def seconds(a, b):
        return (pivot[b] - pivot[a]).dt.total_seconds()
    return pd.DataFrame({
        "geracao_publicacao_s": seconds("Leitura gerada", "Publicada no Kafka"),
        "transito_kafka_s": seconds("Publicada no Kafka", "Recebida pelo consumer"),
        "validacao_gravacao_s": seconds("Recebida pelo consumer", "Gravada na trusted"),
        "espera_micro_lote_s": seconds("Gravada na trusted", "Consolidada na refined"),
        "decisao_s": seconds("Consolidada na refined", "Decisao gerada"),
        "lead_time_total_s": seconds("Leitura gerada", "Decisao gerada"),
    })


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_variants(table: pd.DataFrame, path: Path) -> None:
    top = table.head(8).copy()
    top["rotulo"] = [f"V{i + 1}: {r.resultado} ({r.etapas} etapas)" for i, r in enumerate(top.itertuples())]
    top = top.iloc[::-1]
    labels = top["rotulo"].tolist()
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(top) + 1.4))
    bars = ax.barh(labels, top["casos"], color=ACCENT, height=0.6)
    for bar, pct in zip(bars, top["percentual"], strict=True):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {int(bar.get_width())} ({pct}%)",
                va="center", color=INK, fontsize=9)
    _style(ax, "Variantes do processo (caminhos mais frequentes)", "Casos (leituras)")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


REWORK = {"Reenvio (duplicata)", "Descartada (duplicada)"}


def relevant(table: pd.DataFrame, cases: int) -> pd.DataFrame:
    """Gargalos do fluxo principal: ignora transicoes raras (< 2% dos casos) e o retrabalho por reenvio,
    cuja espera reflete o intervalo do simulador e e analisado a parte."""
    frequent = table["ocorrencias"] >= max(5, cases * 0.02)
    rework = table["de"].isin(REWORK) | table["para"].isin(REWORK)
    return table[frequent & ~rework]


def plot_bottlenecks(table: pd.DataFrame, path: Path) -> None:
    top = table.sort_values("mediana_s", ascending=False).head(8).iloc[::-1]
    labels = [f"{r.de} -> {r.para}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(top) + 1.4))
    bars = ax.barh(labels, top["mediana_s"], color=ACCENT, height=0.6)
    for bar, p95 in zip(bars, top["p95_s"], strict=True):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {bar.get_width():.2f} s (p95 {p95:.2f} s)", va="center", color=INK, fontsize=9)
    _style(ax, "Tempo de espera entre etapas (mediana)", "Segundos")
    ax.margins(x=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lead_time(phases: pd.DataFrame, path: Path) -> None:
    labels = {"geracao_publicacao_s": "Geracao -> publicacao", "transito_kafka_s": "Transito no Kafka",
              "validacao_gravacao_s": "Validacao + gravacao", "espera_micro_lote_s": "Espera do micro-lote",
              "decisao_s": "Decisao"}
    means = phases[list(labels)].mean()
    share = means / means.sum() * 100
    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = means.sort_values().index
    bars = ax.barh([labels[c] for c in order], means[order], color=ACCENT, height=0.6)
    for bar, column in zip(bars, order, strict=True):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {means[column]:.3f} s ({share[column]:.0f}%)", va="center", color=INK, fontsize=9)
    _style(ax, f"Composicao do lead time medio ({phases['lead_time_total_s'].mean():.2f} s ate a decisao)",
           "Segundos (media)")
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def discover_models(frame: pd.DataFrame, out: Path) -> list[str]:
    """Mapas de processo com pm4py (requer Graphviz instalado)."""
    try:
        import pm4py
    except ImportError:
        return ["pm4py nao instalado: modelos graficos nao gerados"]
    log = pm4py.format_dataframe(frame[["case_id", "activity", "timestamp"]].copy(), case_id="case_id",
                                 activity_key="activity", timestamp_key="timestamp")
    notes = []
    try:
        dfg, start, end = pm4py.discover_dfg(log)
        pm4py.save_vis_dfg(dfg, start, end, str(out / "dfg_frequencia.png"), rankdir="TB")
        perf, start, end = pm4py.discover_performance_dfg(log)
        pm4py.save_vis_performance_dfg(perf, start, end, str(out / "dfg_performance.png"), rankdir="TB")
        bpmn = pm4py.discover_bpmn_inductive(log, noise_threshold=0.0)
        pm4py.save_vis_bpmn(bpmn, str(out / "bpmn_processo.png"))
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
        fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
        notes.append(f"fitness_modelo_filtrado={fitness.get('log_fitness', 0):.3f}")
    except Exception as exc:  # Graphviz ausente ou erro de renderizacao nao impede o relatorio
        notes.append(f"falha ao gerar modelos graficos: {exc}")
    return notes


def build_summary(frame, table, trans, phases) -> dict:
    cases = frame.groupby("case_id")
    traces = cases["activity"].agg(tuple)
    received = frame[frame["activity"] == "Recebida pelo consumer"]
    decisions = frame[frame["activity"] == "Decisao gerada"].drop_duplicates("case_id")
    alerts = frame[frame["activity"] == "Alerta emitido"].drop_duplicates("case_id")
    scenario = frame.drop_duplicates("case_id").set_index("case_id")["scenario"]
    by_scenario = (pd.DataFrame({"cenario": scenario, "alerta": scenario.index.isin(alerts["case_id"]),
                                 "decidida": scenario.index.isin(decisions["case_id"])})
                   .query("decidida").groupby("cenario")["alerta"].agg(["count", "mean"]))
    field = frame.drop_duplicates("case_id").set_index("case_id")["field_id"]
    by_field = pd.DataFrame({"talhao": field, "alerta": field.index.isin(alerts["case_id"])}).groupby(
        "talhao")["alerta"].agg(["count", "mean"])
    outcome = traces.map(classify_trace).value_counts()
    validated = frame[frame["activity"] == "Validada"].drop_duplicates("case_id")
    flags = validated.get("out_of_range_fields", pd.Series(dtype=str)).fillna("").astype(str)
    risky = set(validated.loc[flags.str.len() > 0, "case_id"])
    masked = risky - set(alerts["case_id"])
    first_seen = frame.groupby(["case_id", "activity"])["timestamp"].min().unstack()
    out_of_order = 0
    if {"Publicada no Kafka", "Recebida pelo consumer"} <= set(first_seen.columns):
        out_of_order = int((first_seen["Recebida pelo consumer"] < first_seen["Publicada no Kafka"]).sum())
    summary = {
        "periodo": {"inicio": str(frame["timestamp"].min()), "fim": str(frame["timestamp"].max())},
        "eventos": int(len(frame)), "casos": int(frame["case_id"].nunique()),
        "atividades": sorted(frame["activity"].unique()), "variantes": int(len(table)),
        "caminho_feliz_pct": round(float(traces.map(lambda t: t[:len(HAPPY_PATH)] == HAPPY_PATH).mean() * 100), 1),
        "resultados": outcome.to_dict(),
        "reenvios": int((received.groupby("case_id").size() > 1).sum()),
        "taxa_alerta_pct": round(len(alerts) / max(len(decisions), 1) * 100, 1),
        "leituras_fora_da_faixa": len(risky),
        "alertas_mascarados": len(masked),
        "alertas_mascarados_pct": round(len(masked) / max(len(risky), 1) * 100, 1),
        "recebidas_antes_do_ack": out_of_order,
        "casos_incompletos": int(outcome.get("Incompleto", 0)),
        "alerta_por_cenario": {k: {"casos": int(v["count"]), "alerta_pct": round(v["mean"] * 100, 1)}
                               for k, v in by_scenario.iterrows()},
        "alerta_por_talhao": {k: {"casos": int(v["count"]), "alerta_pct": round(v["mean"] * 100, 1)}
                              for k, v in by_field.iterrows()},
        "gargalos": relevant(trans, frame["case_id"].nunique()).sort_values("mediana_s", ascending=False).head(3)[
            ["de", "para", "mediana_s", "p95_s", "ocorrencias"]].to_dict("records"),
    }
    if not phases.empty:
        means = phases.mean()
        total = means.drop("lead_time_total_s").sum()
        summary["lead_time"] = {
            "media_s": round(float(means["lead_time_total_s"]), 3),
            "mediana_s": round(float(phases["lead_time_total_s"].median()), 3),
            "p95_s": round(float(phases["lead_time_total_s"].quantile(0.95)), 3),
            "participacao_pct": {k: round(float(v / total * 100), 1)
                                 for k, v in means.drop("lead_time_total_s").items()},
        }
    return summary


def write_report(summary: dict, table: pd.DataFrame, path: Path, notes: list[str]) -> None:
    lt = summary.get("lead_time", {})
    share = lt.get("participacao_pct", {})
    bottleneck = summary["gargalos"][0] if summary["gargalos"] else None
    lines = [
        "# Analise de processo - AgroSmart", "",
        f"Periodo analisado: {summary['periodo']['inicio']} a {summary['periodo']['fim']} (UTC).",
        f"{summary['eventos']} eventos, {summary['casos']} casos (leituras), {len(summary['atividades'])} atividades "
        f"e {summary['variantes']} variantes.", "",
        "## Resultado dos casos", "",
        *[f"- {k}: {v} casos" for k, v in summary["resultados"].items()],
        f"- Caminho feliz completo (gerada -> decisao): {summary['caminho_feliz_pct']}% dos casos",
        f"- Leituras recebidas mais de uma vez (retrabalho por reenvio): {summary['reenvios']}",
        f"- Leituras aceitas fora da faixa operacional: {summary['leituras_fora_da_faixa']}, das quais "
        f"{summary['alertas_mascarados']} ({summary['alertas_mascarados_pct']}%) nao geraram alerta",
        f"- Leituras recebidas pelo consumer antes de o producer registrar a confirmacao (ack) do Kafka: "
        f"{summary['recebidas_antes_do_ack']} (variacao por concorrencia, nao erro)",
        f"- Casos incompletos (ainda em processamento no fim da janela analisada): {summary['casos_incompletos']}", "",
        "## Variantes mais frequentes", "",
        "| # | Resultado | Etapas | Casos | % |", "|---|---|---|---|---|",
        *[f"| V{i + 1} | {r.resultado} | {r.etapas} | {r.casos} | {r.percentual}% |"
          for i, r in enumerate(table.head(8).itertuples())], "",
        "## Gargalos (maior tempo de espera mediano entre etapas)", "",
        *[f"- {g['de']} -> {g['para']}: mediana {g['mediana_s']:.3f} s, p95 {g['p95_s']:.3f} s "
          f"({g['ocorrencias']} ocorrencias)" for g in summary["gargalos"]], "",
    ]
    if lt:
        lines += ["## Lead time ate a decisao", "",
                  f"Media {lt['media_s']} s, mediana {lt['mediana_s']} s, p95 {lt['p95_s']} s.",
                  *[f"- {k.replace('_s', '').replace('_', ' ')}: {v}% do tempo" for k, v in share.items()], ""]
    lines += ["## Variacao por cenario e talhao (alerta por leitura)", "",
              *[f"- Cenario {k}: {v['casos']} decisoes, {v['alerta_pct']}% com alerta"
                for k, v in summary["alerta_por_cenario"].items()],
              *[f"- {k}: {v['casos']} leituras, {v['alerta_pct']}% com alerta"
                for k, v in summary["alerta_por_talhao"].items()], "",
              "## Oportunidades de melhoria (geradas a partir dos indicadores)", ""]
    if summary["alertas_mascarados"]:
        lines.append(f"- {summary['alertas_mascarados']} leituras fora da faixa ({summary['alertas_mascarados_pct']}%) "
                     "nao viraram alerta: outra leitura mais recente do mesmo talhao entrou no mesmo micro-lote e a "
                     "decisao considera apenas a ultima. Avaliar o risco de cada leitura antes de consolidar.")
    if share and max(share, key=share.get) == "espera_micro_lote_s":
        lines.append(f"- O micro-lote responde por {share['espera_micro_lote_s']}% do lead time: criar um caminho "
                     "rapido que consolide imediatamente leituras fora da faixa (risco medio/alto), mantendo o lote "
                     "para leituras normais.")
    elif bottleneck:
        lines.append(f"- Principal espera em {bottleneck['de']} -> {bottleneck['para']}: "
                     "priorizar otimizacao dessa etapa.")
    results = summary["resultados"]
    rejected = results.get("Rejeitada na validacao", 0) + results.get("Payload invalido", 0)
    if rejected:
        lines.append(f"- {rejected} leituras rejeitadas: validar na borda (firmware/gateway) e alertar sensores com "
                     "falhas repetidas, evitando trafego inutil no Kafka.")
    if summary["reenvios"]:
        lines.append(f"- {summary['reenvios']} reenvios detectados: habilitar producer idempotente "
                     "(enable_idempotence) para reduzir duplicatas na origem; a deduplicacao atual protege a trusted.")
    lines.append(f"- Taxa de alerta geral de {summary['taxa_alerta_pct']}%: revisar thresholds por talhao para evitar "
                 "fadiga de alertas.")
    if notes:
        lines += ["", "## Observacoes tecnicas", "", *[f"- {n}" for n in notes]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-log", type=Path, default=PROJECT_ROOT / "output/process_mining/event_log.csv")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.event_log.parent
    out.mkdir(parents=True, exist_ok=True)
    frame = load_event_log(args.event_log)
    table = variants(frame)
    trans = transitions(frame)
    phases = lead_times(frame)
    table.to_csv(out / "variantes.csv", index=False)
    trans.to_csv(out / "transicoes.csv", index=False)
    plot_variants(table, out / "variantes.png")
    plot_bottlenecks(relevant(trans, frame["case_id"].nunique()), out / "gargalos.png")
    if not phases.empty:
        phases.to_csv(out / "lead_time_por_caso.csv")
        plot_lead_time(phases, out / "lead_time.png")
    notes = discover_models(frame, out)
    summary = build_summary(frame, table, trans, phases)
    summary["observacoes"] = notes
    (out / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(summary, table, out / "analise_processo.md", notes)
    print(f"Analise gerada em {out}")


if __name__ == "__main__":
    main()
