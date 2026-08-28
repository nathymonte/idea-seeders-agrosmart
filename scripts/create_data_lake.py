import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.data_lake import ensure_data_lake


if __name__ == "__main__":
    base_path = ensure_data_lake()
    print(f"Data Lake criado em: {base_path}")
