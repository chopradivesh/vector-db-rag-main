"""Configuration read from environment variables (and an optional .env file).

No API keys are needed: Ollama runs locally. These values were hardcoded in
the C++ version; here they can be changed without editing code.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory, if present

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))  # the web UI expects 8080
DEMO_DIMS = 16
