"""Text -> 16-D demo vector using keyword matching.

Ported from the `textToEmbedding` function in the original index.html (the
web UI still runs its own JavaScript copy). This lets the Python examples
search the demo database by typing words like "binary tree".

BE HONEST ABOUT WHAT THIS IS: a hand-written keyword -> topic mapping, not a
learned embedding model. It only "understands" the ~80 keywords below. The
real embeddings in this project come from Ollama (see src/llm/ollama_client.py).

Algorithm:
  1. For every word, add 0.35 to each topic that has a matching keyword
     (word contains the keyword, or the keyword starts with the word).
  2. Normalise the scores so the best topic is ~0.88.
  3. Write the score into that topic's 4 dimensions (with small variations),
     leaving every other dimension at a 0.08 baseline.
Time: O(words · keywords).
"""

import random

KEYWORDS: dict[str, list[str]] = {
    "cs": ["algorithm", "data", "tree", "graph", "array", "linked", "hash", "stack", "queue", "sort",
           "binary", "dynamic", "programming", "recursion", "complexity", "pointer", "node", "search",
           "insert", "bfs", "dfs", "heap", "trie"],
    "math": ["calculus", "matrix", "probability", "theorem", "integral", "derivative", "linear", "algebra",
             "equation", "function", "prime", "modular", "combinatorics", "permutation", "eigenvalue",
             "statistics", "proof"],
    "food": ["food", "pizza", "sushi", "ramen", "pasta", "recipe", "cook", "eat", "restaurant", "dish",
             "ingredient", "flavor", "spice", "noodle", "bread", "croissant", "taco", "fish", "rice", "soup"],
    "sports": ["sport", "basketball", "football", "tennis", "chess", "swim", "game", "play", "score", "team",
               "athlete", "competition", "match", "tournament", "olympic", "dribble", "tackle", "serve"],
}
# Which block of 4 dimensions belongs to each topic.
TOPIC_OFFSETS = {"cs": 0, "math": 4, "food": 8, "sports": 12}


def text_to_demo_embedding(text: str, jitter: float = 0.04, rng: random.Random | None = None) -> list[float]:
    """jitter=0 gives a deterministic vector (useful for tests)."""
    rng = rng or random.Random()
    scores = {topic: 0.0 for topic in KEYWORDS}
    for word in text.lower().split():
        for topic, keywords in KEYWORDS.items():
            # Prefix match: the word "a" matches "algorithm", "algebra" and "athlete",
            # so very short words can light up several topics (a known quirk).
            if any(kw in word or kw.startswith(word) for kw in keywords):
                scores[topic] += 0.35

    best = max(max(scores.values()), 0.01)

    def noise() -> float:
        return (rng.random() - 0.5) * jitter

    embedding = [0.08] * 16
    for topic, score in scores.items():
        if score < 0.01:
            continue
        base = min(score / best * 0.88, 0.94)
        i = TOPIC_OFFSETS[topic]
        embedding[i] = max(0.05, base + noise())
        embedding[i + 1] = max(0.05, base + noise())
        embedding[i + 2] = max(0.05, base * 0.92 + noise())
        embedding[i + 3] = max(0.05, base * 0.87 + noise())
    return embedding
