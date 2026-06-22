from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # UvA proxy
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5")

DATA_RAW_DIR       = "/Users/grace/thesis_project/data/raw"
DATA_PROCESSED_DIR = "/Users/grace/thesis_project/data/processed"
