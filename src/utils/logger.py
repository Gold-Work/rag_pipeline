import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
os.makedirs(LOGS_PATH, exist_ok=True)

_log_filename = os.path.join(
    LOGS_PATH, f"rag_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


_formatter = JsonFormatter()

_file_handler = logging.FileHandler(_log_filename, encoding="utf-8")
_file_handler.setFormatter(_formatter)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.handlers.clear()
_root.addHandler(_file_handler)
_root.addHandler(_console_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
