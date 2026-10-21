"""Plain data records stored in the databases.

C++ used `struct VectorItem` and `struct DocItem`; Python dataclasses are the
closest equivalent — they auto-generate __init__ and __repr__ for us.
"""

from dataclasses import dataclass


@dataclass
class VectorItem:
    """One vector plus its metadata. Every index stores these."""

    id: int
    metadata: str  # human-readable label, e.g. "Sushi: vinegared rice..."
    category: str  # "cs", "math", "food", "sports", "doc", ...
    embedding: list[float]


@dataclass
class DocItem:
    """One chunk of a user document, with its real (Ollama) embedding."""

    id: int
    title: str
    text: str
    embedding: list[float]
