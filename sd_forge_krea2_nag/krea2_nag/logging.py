from __future__ import annotations

PREFIX = "[Krea2 NAG]"


def _emit(level: str, message: str) -> None:
    print(f"{PREFIX} {level}: {message}")


def debug(enabled: bool, message: str) -> None:
    if enabled:
        _emit("DEBUG", message)


def info(message: str) -> None:
    _emit("INFO", message)


def warn(message: str) -> None:
    _emit("WARN", message)
