from __future__ import annotations

import math
import os
from functools import cache
from pathlib import Path
from typing import Any

from services.paths import PROJECT_ROOT

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "config/thresholds.yaml"
THRESHOLDS_ENV_VAR = "AGROSMART_THRESHOLDS_PATH"


class ThresholdConfigError(ValueError):
    pass


def get_threshold_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or os.getenv(THRESHOLDS_ENV_VAR, DEFAULT_THRESHOLDS_PATH))
    return _load_threshold_config(str(path))


def clear_threshold_config_cache() -> None:
    _load_threshold_config.cache_clear()


@cache
def _load_threshold_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ThresholdConfigError(
            f"Arquivo de configuracao de thresholds nao encontrado: {path}. "
            f"Defina {THRESHOLDS_ENV_VAR} ou crie config/thresholds.yaml."
        )

    with path.open("r", encoding="utf-8") as file:
        if yaml is not None:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise ThresholdConfigError(f"YAML invalido em {path}: {exc}") from exc
        else:
            data = _parse_simple_threshold_yaml(file.read())

    _validate_config(data, path)
    return data


def _validate_config(data: Any, path: Path) -> None:
    if not isinstance(data, dict):
        raise ThresholdConfigError(f"Configuracao invalida em {path}: raiz deve ser um objeto YAML.")

    fields = data.get("sensor_fields")
    if not isinstance(fields, dict) or not fields:
        raise ThresholdConfigError(f"Configuracao invalida em {path}: sensor_fields deve existir e conter campos.")

    for field_name, field_config in fields.items():
        if not isinstance(field_config, dict):
            raise ThresholdConfigError(f"Configuracao invalida em {path}: {field_name} deve ser um objeto.")

        allowed_range = field_config.get("allowed_range")
        if not isinstance(allowed_range, dict):
            raise ThresholdConfigError(f"Configuracao invalida em {path}: {field_name}.allowed_range e obrigatorio.")
        _validate_range(allowed_range, path, f"{field_name}.allowed_range")

        expected_range = field_config.get("expected_range")
        if expected_range is not None:
            if not isinstance(expected_range, dict):
                raise ThresholdConfigError(
                    f"Configuracao invalida em {path}: {field_name}.expected_range deve ser objeto.")
            _validate_range(expected_range, path, f"{field_name}.expected_range")


def _validate_range(range_config: dict[str, Any], path: Path, label: str) -> None:
    minimum = range_config.get("minimum")
    maximum = range_config.get("maximum")

    if (not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float))
            or isinstance(minimum, bool) or isinstance(maximum, bool)
            or not math.isfinite(minimum) or not math.isfinite(maximum)):
        raise ThresholdConfigError(f"Configuracao invalida em {path}: {label} deve ter minimum e maximum numericos.")
    if minimum > maximum:
        raise ThresholdConfigError(f"Configuracao invalida em {path}: {label}.minimum nao pode ser maior que maximum.")


def _parse_simple_threshold_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line:
            raise ThresholdConfigError("YAML invalido: use espacos, nao tabulacoes.")

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ThresholdConfigError(f"YAML invalido: linha sem chave: {line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")
