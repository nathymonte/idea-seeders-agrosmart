import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image

# TensorFlow / Keras
from tensorflow.keras.models import load_model


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels(labels_path: Path) -> list[str]:
    labels: list[str] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            label = parts[1] if len(parts) == 2 else parts[0]
            labels.append(label)
    if not labels:
        raise ValueError(f"Nenhum label encontrado em {labels_path}")
    return labels


def preprocess_image(image_path: Path, size: int = 224) -> np.ndarray:
    """
    Pipeline padrão para Teachable Machine (Keras):
    - abre imagem
    - converte para RGB
    - resize 224x224
    - normaliza para [-1, 1] (muito comum no Teachable)
    - adiciona batch dimension (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img).astype(np.float32)

    # normalização típica do Teachable Machine:
    # de [0,255] -> [-1,1]
    arr = (arr / 127.5) - 1.0

    # adiciona batch dimension
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_folder(
    model_path: Path,
    labels_path: Path,
    input_dir: Path,
    output_path: Path,
    image_size: int = 224,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels não encontrado: {labels_path}")
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Pasta de entrada inválida: {input_dir}")

    labels = load_labels(labels_path)
    LABEL_MAP = {"Health": "Saudavel", "Sick": "Doente"}
    model = tf.keras.models.load_model(model_path, compile=False)

    images = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXTS])

    results = []
    for img_path in images:
        x = preprocess_image(img_path, size=image_size)
        probs = model.predict(x, verbose=0)[0]  # vetor (n_classes,)

        # segurança: garante que tamanho bate com labels
        if len(probs) != len(labels):
            raise ValueError(
                f"Quantidade de probabilidades ({len(probs)}) diferente de labels ({len(labels)})."
            )

        best_idx = int(np.argmax(probs))
        predicted_label = labels[best_idx]
        confidence = float(probs[best_idx])

        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}

        predicted_label_pt = LABEL_MAP.get(predicted_label, predicted_label)
        scores_pt = {LABEL_MAP.get(k, k): v for k, v in scores.items()}

        results.append(
            {
                "image_name": img_path.name,
                "image_path": str(img_path.relative_to(input_dir)),
                "predicted_label": predicted_label,
                "predicted_label_pt": predicted_label_pt,
                "confidence": confidence,
                "scores": scores,
                "scores_pt": scores_pt,
            }
        )

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_file": model_path.name,
            "labels_file": labels_path.name,
            "image_size": image_size,
            "num_images": len(results),
            "classes": labels,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Classifica imagens em lote usando modelo do Teachable Machine.")
    parser.add_argument("--model", default="models/keras_model.h5", help="Caminho do keras_model.h5")
    parser.add_argument("--labels", default="models/labels.txt", help="Caminho do labels.txt")
    parser.add_argument("--in_dir", default="input_images", help="Pasta com imagens para classificar")
    parser.add_argument("--out", default="output/results.json", help="Caminho do JSON de saída")
    parser.add_argument("--size", type=int, default=224, help="Tamanho (224) usado no Teachable Machine")

    args = parser.parse_args()

    predict_folder(
        model_path=Path(args.model),
        labels_path=Path(args.labels),
        input_dir=Path(args.in_dir),
        output_path=Path(args.out),
        image_size=args.size,
    )

    print(f"OK: exportado {args.out}")


if __name__ == "__main__":
    main()