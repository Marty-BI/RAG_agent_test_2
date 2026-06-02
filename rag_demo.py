import numpy as np
import ollama

# --- 1. Our tiny "knowledge base" ---
# These are the documents we want to search over.
documents = [
    "The cat sat on the mat.",
    "Dogs love to play fetch in the park.",
    "Python is a popular programming language.",
    "Machine learning models can understand text.",
]

# --- 2. Embed every document ---
# An embedding turns text into a list of numbers (a vector).
# Similar meanings produce vectors that point in similar directions.
doc_vectors = []
for doc in documents:
    response = ollama.embed(model="nomic-embed-text", input=doc)
    doc_vectors.append(np.array(response["embeddings"][0]))

# --- 3. Peek inside one vector ---
# This shows you what an embedding actually looks like: hundreds of floats.
print(f"Vector length : {len(doc_vectors[0])}")
print(f"First 5 values: {doc_vectors[0][:5]}\n")

# --- 4. Embed the query ---
# We want to find whichever document is most "similar" to this question.
query = "What programming languages are used in AI?"
query_response = ollama.embed(model="nomic-embed-text", input=query)
query_vec = np.array(query_response["embeddings"][0])

# --- 5. Cosine similarity, computed by hand ---
# Imagine two arrows (vectors) in space.
# Cosine similarity measures the ANGLE between them, not their length.
#   • Score =  1.0  → same direction → very similar meaning
#   • Score =  0.0  → perpendicular  → unrelated
#   • Score = -1.0  → opposite       → opposite meaning
#
# Formula:  cos(θ) = (A · B) / (‖A‖ × ‖B‖)
#   A · B   = dot product  (element-wise multiply, then sum)
#   ‖A‖     = magnitude    (square-root of sum of squares)
#
# We divide by magnitudes so that a short sentence and a long sentence
# with the same meaning still score 1.0 — length doesn't matter, only direction.

scores = []
for doc, vec in zip(documents, doc_vectors):
    dot_product = np.dot(query_vec, vec)          # how much they "agree"
    magnitude   = np.linalg.norm(query_vec) * np.linalg.norm(vec)  # scale factor
    similarity  = dot_product / magnitude
    scores.append((similarity, doc))

# --- 6. Print results, best match first ---
scores.sort(reverse=True)
print("Results (most relevant → least relevant):")
for rank, (score, doc) in enumerate(scores, start=1):
    print(f"  {rank}. [{score:.4f}]  {doc}")
