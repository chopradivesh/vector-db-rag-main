"""Shared test helpers.

FakeLLM stands in for Ollama so the RAG pipeline can be tested without a
running model: it embeds text as word counts over a small fixed vocabulary
(a tiny bag-of-words model), so texts sharing words get similar vectors.
"""

import random

import pytest

from src.vector_db.models import VectorItem

VOCAB = ["python", "snake", "language", "code", "pizza", "cheese", "tomato", "oven",
         "football", "goal", "team", "match", "planet", "orbit", "star", "space"]


class FakeLLM:
    gen_model = "fake-llm"

    def __init__(self):
        self.prompts: list[str] = []

    def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(sum(w.strip(".,?!") == v for w in words)) for v in VOCAB]

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "fake answer"


@pytest.fixture
def fake_llm():
    return FakeLLM()


def random_items(n: int, dims: int, seed: int = 0) -> list[VectorItem]:
    rng = random.Random(seed)
    return [VectorItem(i, f"item {i}", "rand", [rng.random() for _ in range(dims)]) for i in range(1, n + 1)]
