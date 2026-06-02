import numpy as np
import ollama

# ── Tuning knobs ─────────────────────────────────────────────────────────────
# CHUNK_SIZE : how many characters per chunk.
#   Too large  → each chunk covers many topics; the embedding blurs them together,
#                so a narrow query matches a chunk that's only partly relevant.
#   Too small  → a chunk may not contain enough context for the model to embed
#                a clear meaning; also produces more chunks = more API calls.
CHUNK_SIZE = 200

# OVERLAP : how many characters the end of one chunk shares with the start of
#   the next.  Without overlap, a sentence split across the boundary loses its
#   context in BOTH neighbouring chunks.  Overlap lets each boundary sentence
#   appear fully in at least one chunk.
#   Rule of thumb: 10–20 % of CHUNK_SIZE is usually enough.
OVERLAP = 100

TOP_K = 3   # how many chunks to show in the final answer

# ── 1. Our "document" ────────────────────────────────────────────────────────
# Four clearly different subtopics so we can easily judge retrieval quality.
TEXT = """
Python is a high-level programming language famous for its readable syntax.
It was created by Guido van Rossum and released in 1991. Developers love it
because a few lines of Python can replace dozens of lines in other languages.

Machine learning is a branch of artificial intelligence where computers learn
patterns from data instead of following hand-written rules. Popular frameworks
include TensorFlow and PyTorch, which make it easier to build neural networks.

The solar system contains eight planets orbiting the Sun. The four inner planets
are rocky, while the four outer ones are gas or ice giants. Jupiter is the
largest planet and has a famous storm called the Great Red Spot.

A balanced diet includes proteins, carbohydrates, healthy fats, vitamins, and
minerals. Proteins are essential for building muscle tissue. Carbohydrates
provide the main energy source for the brain and body during physical activity.
""".strip()

# ── 2. Split into overlapping character-based chunks ─────────────────────────
def make_chunks(text, chunk_size, overlap):
    """
    Slide a window of `chunk_size` characters across `text`, advancing by
    (chunk_size - overlap) each step so consecutive chunks share `overlap`
    characters at their boundary.
    """
    step = chunk_size - overlap   # how far we move the window each iteration
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

chunks = make_chunks(TEXT, CHUNK_SIZE, OVERLAP)

# ── 3. Print every chunk so we can see the cuts ──────────────────────────────
print(f"Total characters: {len(TEXT)}")
print(f"Chunk size: {CHUNK_SIZE}  |  Overlap: {OVERLAP}  |  Chunks produced: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"── Chunk {i} ({len(chunk)} chars) ──────────────────")
    print(chunk)
    print()

# ── 4. Embed every chunk ─────────────────────────────────────────────────────
print("Embedding chunks… ", end="", flush=True)
chunk_vectors = []
for chunk in chunks:
    response = ollama.embed(model="nomic-embed-text", input=chunk)
    chunk_vectors.append(np.array(response["embeddings"][0]))
print("done.\n")

# ── 5. Cosine similarity (same formula as before) ────────────────────────────
# cos(θ) = (A · B) / (‖A‖ × ‖B‖)
# Measures the angle between two vectors; 1.0 = same direction, 0.0 = unrelated.
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ── Query → embed → rank ─────────────────────────────────────────────────────
query = "Which coding tools do data scientists rely on?"
query_response = ollama.embed(model="nomic-embed-text", input=query)
query_vec = np.array(query_response["embeddings"][0])

scores = [
    (cosine_similarity(query_vec, vec), i, chunk)
    for i, (chunk, vec) in enumerate(zip(chunks, chunk_vectors))
]
scores.sort(reverse=True)   # highest similarity first

print(f'Query: "{query}"\n')
print(f"Top {TOP_K} chunks:\n")
for rank, (score, idx, chunk) in enumerate(scores[:TOP_K], start=1):
    print(f"  #{rank}  chunk {idx}  [score {score:.4f}]")
    print(f"  {chunk!r}\n")
