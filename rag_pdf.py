import numpy as np
import ollama
from pypdf import PdfReader

# ── Config ───────────────────────────────────────────────────────────────────
PDF_PATH   = "sample.pdf"   # <-- change this to your PDF file path
CHUNK_SIZE = 500            # larger than before — real sentences need more room
OVERLAP    = 80
TOP_K      = 3
EMBED_MODEL = "nomic-embed-text"

# ── 1. Load the PDF ───────────────────────────────────────────────────────────
def load_pdf(path):
    """
    Extract all text from a PDF and return it as one big string.

    pypdf reads each page and calls .extract_text(), which tries to reassemble
    the text from the PDF's internal drawing instructions.  PDF was designed for
    printing, NOT for text storage, so extraction is often imperfect:

      • Broken words: a word split with a hyphen across a line may become two
        tokens ("computa-\ntion") or get merged without a space ("computation").
      • Headers / footers: page numbers, chapter titles, and running headers are
        extracted inline — they'll appear as noise in the middle of your chunks.
      • Irregular spacing: columns, tables, and multi-column layouts produce
        text in the wrong reading order, or with large gaps (\xa0, multiple
        spaces) that look like paragraph breaks but aren't.
      • Garbled characters: some PDFs store text as glyph IDs rather than
        Unicode; those pages come out as question marks or random symbols.

    The diagnostics printed below are your first check of how clean the text is.
    """
    reader = PdfReader(path)
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""   # returns None for image-only pages
        pages.append(text)
    return pages, "\n".join(pages)         # also return pages for page-count diagnostic

# ── 2. Extract and print diagnostics ─────────────────────────────────────────
pages, full_text = load_pdf(PDF_PATH)

print("=" * 60)
print("PDF DIAGNOSTICS  (inspect this before trusting your chunks)")
print("=" * 60)

# Total size tells you whether the PDF is mostly text or mostly images.
print(f"Pages extracted : {len(pages)}")
print(f"Total characters: {len(full_text)}")

# Page-break points are where "\n" joins appear between pages.
# A high count relative to total chars hints at lots of short pages (headers,
# title pages, blank pages) that will pollute your chunks with noise.
page_breaks = full_text.count("\n")
print(f"Newline count   : {page_breaks}")

# raw repr() shows EXACTLY what was extracted — \n, \xa0, odd spacing, etc.
# Reading this is the fastest way to spot junk before it reaches the embedder.
print(f"\nFirst 500 characters (raw repr so whitespace/junk is visible):")
print(repr(full_text[:500]))
print("=" * 60, "\n")

# ── Reused helpers (identical to previous scripts) ────────────────────────────
def make_chunks(text, chunk_size, overlap):
    # Naive sliding window over raw characters — intentionally unchanged so we
    # can observe how it behaves on real (potentially messy) PDF text.
    step, chunks, start = chunk_size - overlap, [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks

def cosine_similarity(a, b):
    # cos(θ) = (A · B) / (‖A‖ × ‖B‖)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ── 3. Chunk and inspect the first three ─────────────────────────────────────
chunks = make_chunks(full_text, CHUNK_SIZE, OVERLAP)
print(f"Chunks produced : {len(chunks)}  (size={CHUNK_SIZE}, overlap={OVERLAP})\n")

# Printing raw repr reveals mid-word cuts, stray newlines, and header/footer
# noise that the naive splitter doesn't know to avoid.
print("First 3 chunks (raw repr — watch for awkward cuts and PDF artefacts):")
for i in range(min(3, len(chunks))):
    print(f"\n── Chunk {i} ({len(chunks[i])} chars) ──────────────────")
    print(repr(chunks[i]))
print()

# ── 4. Embed all chunks ───────────────────────────────────────────────────────
print("Embedding chunks… ", end="", flush=True)
chunk_vectors = [
    np.array(ollama.embed(model=EMBED_MODEL, input=c)["embeddings"][0])
    for c in chunks
]
print(f"done.\n")

# ── 5. Retrieve top-K for query ───────────────────────────────────────────────
query = "What is the main topic discussed in this document?"   # <-- change me
query_vec = np.array(ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0])

ranked = sorted(
    enumerate(chunks),
    key=lambda ic: cosine_similarity(query_vec, chunk_vectors[ic[0]]),
    reverse=True,
)

print(f'Query: "{query}"\n')
print(f"Top {TOP_K} chunks by cosine similarity:\n")
for rank, (idx, chunk) in enumerate(ranked[:TOP_K], start=1):
    score = cosine_similarity(query_vec, chunk_vectors[idx])
    print(f"  #{rank}  chunk {idx}  [score {score:.4f}]")
    print(f"  {chunk!r}\n")
