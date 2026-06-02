import numpy as np
import ollama

# ── Config ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 200
OVERLAP    = 40
TOP_K      = 3
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "llama3.2"

# ── 1. Document, chunking, embedding (recap from previous step) ──────────────
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

def make_chunks(text, chunk_size, overlap):
    step, chunks, start = chunk_size - overlap, [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks

def cosine_similarity(a, b):
    # cos(θ) = (A · B) / (‖A‖ × ‖B‖)  — 1.0 = identical direction, 0.0 = unrelated
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

chunks = make_chunks(TEXT, CHUNK_SIZE, OVERLAP)

print("Embedding chunks… ", end="", flush=True)
chunk_vectors = [
    np.array(ollama.embed(model=EMBED_MODEL, input=c)["embeddings"][0])
    for c in chunks
]
print(f"done ({len(chunks)} chunks).\n")

# ── Retrieve top-K chunks for the query ──────────────────────────────────────
query = "Which coding tools do data scientists rely on?"
query_vec = np.array(ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0])

ranked = sorted(
    enumerate(chunks),
    key=lambda ic: cosine_similarity(query_vec, chunk_vectors[ic[0]]),
    reverse=True,
)
top_chunks = [chunk for _, chunk in ranked[:TOP_K]]

# ── 2. Build the prompt ───────────────────────────────────────────────────────
# We tell the model to answer ONLY from the supplied context for two reasons:
#   a) Grounding: the model can't hallucinate facts that aren't in the context.
#   b) Verifiability: every claim in the answer must trace back to a chunk we
#      can show the user, making the system auditable.
# "Say so if the answer isn't here" is equally important — it turns a confident
# wrong answer into an honest "I don't know", which is far more useful.

context_block = "\n\n".join(
    f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(top_chunks)
)

prompt = f"""You are a helpful assistant. Answer the question below using ONLY
the context passages provided. Do not use any outside knowledge. If the answer
cannot be found in the context, say "I don't have enough context to answer that."

{context_block}

Question: {query}""".strip()

# ── 3. Print the full prompt so the learner can see what the model receives ───
print("=" * 60)
print("PROMPT SENT TO MODEL")
print("=" * 60)
print(prompt)
print("=" * 60, "\n")

# ── 4. RAG answer — model sees the retrieved context ─────────────────────────
rag_response = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": prompt}],
)
print("ANSWER  (with RAG context)")
print("-" * 40)
print(rag_response["message"]["content"])
print()

# ── 5. Bare answer — same question, zero context ─────────────────────────────
# This contrast is the whole point of RAG:
#   Without context the model draws on its training data and may produce a
#   fluent but generic (or wrong) answer.  With context it is anchored to OUR
#   document — a document that could be private, up-to-date, or domain-specific
#   in ways the model's training never covered.
bare_response = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": query}],
)
print("ANSWER  (no context — bare question)")
print("-" * 40)
print(bare_response["message"]["content"])
