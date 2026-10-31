"""Healthcheck usado pelo Docker: verifica heartbeat de um componente ou a URL do dashboard.

Uso:
    python -m scripts.healthcheck consumer --max-age 60
    python -m scripts.healthcheck dashboard --url http://localhost:8501/_stcore/health
"""
import argparse
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.observability import read_status


def check_heartbeat(component: str, max_age: float) -> tuple[bool, str]:
    status = read_status(component)
    if not status:
        return False, f"{component}: sem heartbeat"
    beat = datetime.fromisoformat(status["heartbeat_at"].replace("Z", "+00:00"))
    age = (datetime.now(UTC) - beat).total_seconds()
    if status.get("state") == "error":
        return False, f"{component}: estado de erro"
    return age <= max_age, f"{component}: heartbeat ha {age:.0f}s (estado {status.get('state')})"


def check_url(url: str) -> tuple[bool, str]:
    if not url.startswith(("http://localhost", "http://127.0.0.1")):
        return False, "apenas URLs locais sao verificadas"
    try:
        # URL restrita a localhost na validacao acima
        with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
            return response.status == 200, f"{url}: HTTP {response.status}"
    except OSError as exc:
        return False, f"{url}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("component")
    parser.add_argument("--max-age", type=float, default=60)
    parser.add_argument("--url")
    args = parser.parse_args()
    healthy, message = check_url(args.url) if args.url else check_heartbeat(args.component, args.max_age)
    print(message)
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
