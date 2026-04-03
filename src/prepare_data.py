import json
import random
from pathlib import Path

import pandas as pd

INPUT_JSON = Path("output/results.json")
OUTPUT_CSV = Path("output/classificacoes.csv")

random.seed(42)

LOCALIDADES = ["Talhão A", "Talhão B", "Talhão C"]
DATAS = [
    "2026-03-03",
    "2026-03-04",
    "2026-03-05",
    "2026-03-06",
    "2026-03-07",
    "2026-03-08",
]
ANOMALIAS = ["Mancha foliar", "Ferrugem", "Oídio", "Praga mastigadora"]


def infer_actual_label(image_name: str) -> str:
    name = image_name.lower().strip()
    if name.startswith("health"):
        return "Health"
    if name.startswith("sick"):
        return "Sick"
    return "Unknown"


def translate_label(label: str) -> str:
    mapping = {
        "Health": "Saudável",
        "Sick": "Doente",
        "Unknown": "Desconhecida",
    }
    return mapping.get(label, label)


def simulate_anomaly(actual_or_predicted_label: str) -> str:
    if actual_or_predicted_label == "Health":
        return "Nenhuma"
    if actual_or_predicted_label == "Sick":
        return random.choice(ANOMALIAS)
    return "Não informada"


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    results = data.get("results", [])

    for i, item in enumerate(results):
        image_name = item["image_name"]
        predicted_label = item["predicted_label"]
        actual_label = infer_actual_label(image_name)

        row = {
            "image_name": image_name,
            "predicted_label": predicted_label,
            "predicted_label_pt": translate_label(predicted_label),
            "confidence": round(float(item["confidence"]) * 100, 2),
            "score_health": round(float(item["scores"]["Health"]) * 100, 2),
            "score_sick": round(float(item["scores"]["Sick"]) * 100, 2),
            "actual_label": actual_label,
            "actual_label_pt": translate_label(actual_label),
            "is_correct": predicted_label == actual_label,
            "data_analise": DATAS[i % len(DATAS)],
            "localidade": random.choice(LOCALIDADES),
            "anomalia": simulate_anomaly(predicted_label),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"CSV gerado com sucesso em: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()