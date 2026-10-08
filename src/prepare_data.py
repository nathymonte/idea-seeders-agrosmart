import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.paths import PROJECT_ROOT, IMAGE_CSV_PATH, validate_field_id
from services.data_lake import parse_datetime, recompute_field_status

CSV_FIELDS = ["image_name", "image_path", "predicted_label", "predicted_label_pt", "confidence",
              "score_health", "score_sick", "actual_label", "is_correct", "data_analise",
              "field_id", "localidade", "anomalia", "metadata_source", "label_source"]


def infer_actual_label(image_name: str) -> str:
    name = image_name.lower().strip()
    return "Health" if name.startswith("health") else "Sick" if name.startswith("sick") else "Unknown"


def translate_label(label: str) -> str:
    return {"Health": "Saudavel", "Sick": "Indicio de doenca", "Unknown": "Desconhecida"}.get(label, label)


def prepare_rows(data: dict, metadata: dict[str, dict] | None = None) -> list[dict]:
    rows = []
    for item in data.get("results", []):
        image_name = item["image_name"]
        image_path = item.get("image_path", image_name)
        info = (metadata or {}).get(image_path, {})
        field_id = info.get("field_id", "")
        if field_id:
            validate_field_id(field_id)
        timestamp = info.get("captured_at") or data.get("meta", {}).get("generated_at", "")
        if timestamp:
            timestamp = parse_datetime(timestamp).isoformat()
        actual_label = info.get("actual_label") or infer_actual_label(image_name)
        predicted = item["predicted_label"]
        rows.append({
            "image_name": image_name, "image_path": image_path,
            "predicted_label": predicted, "predicted_label_pt": translate_label(predicted),
            "confidence": round(float(item["confidence"]) * 100, 2),
            "score_health": round(float(item["scores"]["Health"]) * 100, 2),
            "score_sick": round(float(item["scores"]["Sick"]) * 100, 2),
            "actual_label": actual_label,
            "is_correct": predicted == actual_label if actual_label != "Unknown" else "",
            "data_analise": timestamp, "field_id": field_id, "localidade": field_id or "Sem associacao",
            "anomalia": "Nao diagnosticada" if predicted == "Sick" else "Nenhuma",
            "metadata_source": info.get("metadata_source", "manifest" if info else "unassigned"),
            "label_source": "manifest" if info.get("actual_label") else "filename_hint",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepara previsoes e associa metadados explicitos de imagens.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "output/results.json")
    parser.add_argument("--out", type=Path, default=IMAGE_CSV_PATH)
    parser.add_argument("--metadata", type=Path, default=os.getenv("AGROSMART_IMAGE_METADATA"))
    args = parser.parse_args()
    metadata = {}
    if args.metadata:
        with args.metadata.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                key = row.get("image_path") or row.get("image_name")
                if not key or key in metadata:
                    raise ValueError("Manifest deve ter image_path unico para cada imagem")
                metadata[key] = row
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(prepare_rows(data, metadata))
    recompute_field_status(output_csv=args.out)
    print(f"CSV preparado: {args.out}")


if __name__ == "__main__":
    main()
