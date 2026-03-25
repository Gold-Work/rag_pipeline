import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
os.makedirs(LOGS_PATH, exist_ok=True)

log_filename = os.path.join(LOGS_PATH, f"ingestion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)