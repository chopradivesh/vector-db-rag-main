"""Split long text into overlapping word windows before embedding.

Why chunk at all?
  * Embedding models have an input limit, and one vector for a whole book
    averages away the details. Small chunks = precise retrieval.
  * Only the relevant chunks go into the LLM prompt, keeping it short.
Why overlap?
  * A sentence cut at a chunk boundary still appears whole in the next chunk.

Input:  text, chunk_words (default 250), overlap_words (default 30)
Output: list of chunk strings. Window starts at 0, 220, 440, ... (step = 250 - 30)
Time/space: O(W), W = number of words.
"""


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    if overlap_words >= chunk_words:
        # Added guard: step would be <= 0 and the loop would never end.
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()  # splits on any whitespace, like C++ `ss >> w`
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]  # short text: keep it exactly as given

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break  # last window already reaches the end; avoid a tiny tail chunk
    return chunks
