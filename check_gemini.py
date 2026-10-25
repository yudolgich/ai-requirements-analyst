from __future__ import annotations

import argparse
import os
import socket
import ssl
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from gemini import (
    DEFAULT_GEMINI_MODEL,
    _safe_error_text,
    _version_key,
    load_text_generation_models,
    probe_text_generation_model,
    select_catalog_model,
)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(
        description="Проверка каталога и одного короткого запроса Gemini."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Вывести каталог, не отправляя запрос генерации.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Тайм-аут короткого запроса в секундах (минимум 10; по умолчанию 12).",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY не задан; значение ключа не требуется выводить.", file=sys.stderr)
        return 2
    if args.timeout < 10:
        parser.error("--timeout не может быть меньше 10 секунд: это ограничение SDK Gemini")

    hostname = "generativelanguage.googleapis.com"
    try:
        with socket.create_connection((hostname, 443), timeout=min(5.0, args.timeout)) as connection:
            with ssl.create_default_context().wrap_socket(
                connection, server_hostname=hostname
            ):
                pass
    except OSError as error:
        print(
            f"HTTPS-подключение к Gemini API не удалось: "
            f"{_safe_error_text(error, api_key)}",
            file=sys.stderr,
        )
        return 1

    try:
        models = load_text_generation_models(api_key, args.timeout)
    except Exception as error:
        print(
            f"Не удалось получить каталог Gemini ({type(error).__name__}): "
            f"{_safe_error_text(error, api_key)}",
            file=sys.stderr,
        )
        return 1

    model_names = sorted(models, key=_version_key, reverse=True)
    print(f"HTTPS-подключение к {hostname}: успешно.")
    print("Доступные модели с поддержкой generateContent:")
    if not model_names:
        print("  API не вернул доступных моделей.")
        return 1
    for name in model_names:
        print(f"  - {name}")

    configured_model = os.getenv("GEMINI_MODEL", "")
    try:
        selected_model = select_catalog_model(models, configured_model)
    except Exception as error:
        print(f"Не удалось выбрать модель: {_safe_error_text(error, api_key)}", file=sys.stderr)
        return 1

    if configured_model and configured_model != selected_model:
        print(
            f"GEMINI_MODEL={configured_model} отсутствует в каталоге; "
            f"для диагностики выбрана безопасная модель по умолчанию {selected_model}."
        )
    elif not configured_model:
        print(f"GEMINI_MODEL не задан; используется модель по умолчанию {DEFAULT_GEMINI_MODEL}.")
    else:
        print(f"Выбранная модель: {selected_model}")

    if args.list_only:
        print("Проверка генерации пропущена (--list-only).")
        return 0

    print(
        f"Отправляю один минимальный запрос; тайм-аут {args.timeout:g} с. "
        "Повторы и переключение модели отключены.",
        flush=True,
    )
    started_at = time.monotonic()
    try:
        probe_text_generation_model(
            api_key,
            selected_model,
            models,
            timeout_seconds=args.timeout,
        )
    except Exception as error:
        print(
            f"Запрос к {selected_model} завершился ошибкой через "
            f"{time.monotonic() - started_at:.1f} с "
            f"({type(error).__name__}): {_safe_error_text(error, api_key)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Минимальный запрос к {selected_model} завершился успешно за "
        f"{time.monotonic() - started_at:.1f} с."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
