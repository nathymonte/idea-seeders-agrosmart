"""Observabilidade do AgroSmart: logs estruturados, heartbeats e eventos de processo.

Cada componente (producer, consumer, dashboard, pipeline de imagens) grava:
- logs JSON em ``logs/<componente>.log`` (rotativos) e no console;
- um heartbeat em ``logs/status/<componente>.json`` com contadores e ultima atividade;
- eventos de processo (campo ``activity``) que alimentam o event log da mineracao de processos.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from services.paths import PROJECT_ROOT


def log_dir() -> Path:
    return Path(os.getenv("AGROSMART_LOG_DIR", str(PROJECT_ROOT / "logs")))


def status_dir() -> Path:
    return log_dir() / "status"

COMPONENTS = ("producer", "consumer", "dashboard", "image-processing")
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_lock = threading.Lock()


def utc_iso(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).isoformat().replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": utc_iso(datetime.fromtimestamp(record.created, UTC)),
            "level": record.levelname,
            "component": record.name.removeprefix("agrosmart."),
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(component: str, log_path: Path | str | None = None) -> logging.Logger:
    """Logger JSON por componente; idempotente (pode ser chamado varias vezes)."""
    logger = logging.getLogger(f"agrosmart.{component}")
    if getattr(logger, "_agrosmart_configured", False):
        return logger
    logger.setLevel(os.getenv("AGROSMART_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    formatter = JsonFormatter()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    directory = Path(log_path or log_dir())
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(directory / f"{component}.log", maxBytes=2_000_000,
                                           backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:  # sem disco gravavel, segue apenas com console
        logger.warning("log em arquivo indisponivel", extra={"error": str(exc)})
    logger._agrosmart_configured = True
    return logger


def log_activity(logger: logging.Logger, activity: str, case_id: str | None, **fields: Any) -> None:
    """Registra um passo do processo (entrada da mineracao de processos)."""
    logger.info(activity, extra={"activity": activity, "case_id": case_id or "sem-event-id", **fields})


def write_heartbeat(component: str, status_path: Path | str | None = None, **metrics: Any) -> Path:
    """Grava atomicamente o estado atual do componente."""
    directory = Path(status_path or status_dir())
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"component": component, "pid": os.getpid(), "heartbeat_at": utc_iso(), **metrics}
    target = directory / f"{component}.json"
    with _lock:
        handle, temporary = tempfile.mkstemp(dir=directory, prefix=f".{component}-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    return target


def read_status(component: str, status_path: Path | str | None = None) -> dict[str, Any] | None:
    path = Path(status_path or status_dir()) / f"{component}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def health_state(status: dict[str, Any] | None, now: datetime | None = None,
                 stale_seconds: float = 30, offline_seconds: float = 120) -> str:
    """Classifica o componente em online / atrasado / offline / finalizado / sem dados."""
    if not status:
        return "sem dados"
    if status.get("state") == "finished":
        return "finalizado"
    try:
        beat = datetime.fromisoformat(str(status["heartbeat_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return "sem dados"
    age = ((now or datetime.now(UTC)) - beat).total_seconds()
    if age <= stale_seconds:
        return "online"
    if age <= offline_seconds:
        return "atrasado"
    return "offline"


def tail_logs(limit: int = 200, log_path: Path | str | None = None,
              components: tuple[str, ...] = COMPONENTS) -> list[dict[str, Any]]:
    """Ultimas linhas JSON de todos os componentes, ordenadas por horario."""
    entries: list[dict[str, Any]] = []
    directory = Path(log_path or log_dir())
    for component in components:
        path = directory / f"{component}.log"
        if not path.exists():
            continue
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - 400_000))
            lines = file.read().decode("utf-8", errors="replace").splitlines()[-limit:]
        for line in lines:
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    entries.sort(key=lambda item: item.get("ts", ""))
    return entries[-limit:]
