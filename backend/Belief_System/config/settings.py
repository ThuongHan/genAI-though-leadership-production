from dotenv import load_dotenv
from pathlib import Path
import os

_BACKEND = Path(__file__).resolve().parent.parent.parent
load_dotenv(_BACKEND / "secrets" / ".env")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # UvA proxy
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5")

DATA_RAW_DIR       = str(_BACKEND / "Belief_System" / "data" / "raw")
DATA_PROCESSED_DIR = str(_BACKEND / "Belief_System" / "data" / "processed")
