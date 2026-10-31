"""Export the existing decision engine's results as a readable agricultural report."""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.data_lake import recompute_field_status
from services.paths import DATA_LAKE_PATH, IMAGE_CSV_PATH, PROJECT_ROOT


def generate_report(destination: Path, base_path=DATA_LAKE_PATH, image_csv=IMAGE_CSV_PATH) -> Path:
    written = recompute_field_status(base_path, image_csv)
    lines = ['# AgroSmart - Relatorio operacional', '',
             f'Gerado em {datetime.now(UTC).isoformat()}', '',
             'Decisoes baseadas na ultima leitura e nas imagens associadas ao talhao nas ultimas 24 horas.',
             'Classificacao de folhas binaria; nao diagnostica doencas especificas.', '']
    for path in written:
        status = json.loads(path.read_text(encoding='utf-8'))
        lines.extend([f"## {status['field_id']}", '', f"Nivel de atencao: {status['attention_level']}",
                      f"Ultima leitura: {status['sensor_summary']['last_reading_at']}",
                      status['recommendation'], ''])
        lines.extend(f'- {reason}' for reason in status['decision_reasons'])
        lines.append('')
    if not written:
        lines.append('Sem leituras validadas para consolidacao.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines), encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera relatorio a partir do motor de decisao integrado.')
    default_name = f'relatorio_agricola_{datetime.now():%Y%m%d_%H%M%S}.md'
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'output' / default_name)
    args = parser.parse_args()
    print(generate_report(args.out))
