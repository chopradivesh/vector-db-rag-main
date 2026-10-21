"""Entry point: start the VectorDB web server (equivalent of the C++ main()).

    python main.py                            -> http://localhost:8080
    uvicorn main:app --host 0.0.0.0 --port 8080   (how Render starts it)
"""

import uvicorn

from src import config
from src.api.server import create_app

# Module-level app so a server command like `uvicorn main:app` can find it.
app = create_app()


def main() -> None:
    ollama = app.state.ollama
    online = ollama.is_available()

    print("=== VectorDB Engine (Python) ===")
    print(f"http://localhost:{config.PORT}")
    print(f"{len(app.state.db)} demo vectors | {config.DEMO_DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama:", "ONLINE" if online else "OFFLINE (install from ollama.com, then: ollama serve)")
    if online:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
