"""
Minimal RAG over Cotap product JSON files.

This replaces PDF extraction (which scrambled our data) with reading the
ORIGINAL structured JSON. The pipeline is now:

    load JSON  ->  curate useful fields  ->  one clean text blob per product
               ->  embed each blob       ->  retrieve top-K for a query
               ->  feed to local chat model (grounded answer)

No frameworks, no vector database. Only `ollama` and `numpy`.
"""

import os
import re
import sys
import json
import glob
import numpy as np
import ollama

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text-v2-moe"   # NOTE: v1 is English-centric; our data is
                                   # Dutch. Good enough to learn; for real
                                   # quality consider 'nomic-embed-text-v2-moe'.
CHAT_MODEL  = "llama3.2"            # set explicitly (don't let it drift to llama3)
TOP_K       = 8

# Point this at a single .json file OR a folder full of them.
DATA_PATH   = r"C:\Users\marbos\Downloads\selected_products"

# A Dutch query, because the product data is Dutch. Try changing it.
QUERY = "Welke vloer is waterbestendig en geschikt voor de badkamer?"

# ── The curation decision (this is the important part) ───────────────────────
# A product JSON has ~100 fields. Most are ERP plumbing (cost_center_code,
# Magazijngroep), internal codes, or image filename hashes (sfeer_0, ...).
# Embedding all of that floods the index with noise. So we CHOOSE the fields a
# customer might actually ask about. This allow-list IS your index design.
USEFUL_FIELDS = [
    "Product type", "Soort tapijt", "Poolmateriaal", "Garentype",
    "Productie methode", "Kleur", "Kleurfamilie",
    "Breedte (cm)", "Poolhoogte (mm)", "Totale dikte (mm)",
    "Gebruiksklasse consumenten", "Woongebruik", "Comfortklasse",
    "Projectgeschikt", "Trapgeschiktheid", "Extra toepassingen",
    "Geschikt voor vloerverwarming", "Geschikt voor (ruimte)",
    "Antislip", "Antistatisch", "Brandclassificatie", "Vloerkleed op maat",
    "Garantie Woongebruik (jaren)",
    "Categorie 1", "Categorie 2", "Categorie 3",
]
# The prose fields are gold for Q&A — keep them last so they read naturally.
DESCRIPTION_FIELDS = ["Cotap artikeltekst", "Brandsites artikeltekst"]


# ── Value cleaning: the JSON is structured but still messy ───────────────────
# Three quirks we handle:
#   1. Some values carry channel tags:      "false[default]"
#   2. Some are JSON encoded INSIDE a string: '["effen"]'  '{"amount":"12","unit":"KG"}'
#   3. Some use underscores as spaces:        "Solution_Dyed_Polyester"
def strip_channel(s):
    """Remove a trailing [tag], e.g. 'trapgeschikt[channel_amb]' -> 'trapgeschikt'."""
    return re.sub(r"\[[^\]]*\]\s*$", "", str(s)).strip()

def normalize(s):
    """Tidy data artefacts so the text reads like language, not codes."""
    s = str(s).replace("_x002D_", "-")
    s = s.replace("__", " ").replace("_", " ")
    return s.strip()

def clean_value(v):
    """Turn any of the messy value shapes into one readable string."""
    # Real JSON list (e.g. channel-tagged variants)
    if isinstance(v, list):
        items, seen = [], []
        for x in v:
            t = normalize(strip_channel(x))
            if t and t not in seen:
                seen.append(t)
        return ", ".join(seen)
    if isinstance(v, str):
        t = v.strip()
        # A JSON array or object stored as a string
        if t.startswith("[") or t.startswith("{"):
            try:
                parsed = json.loads(t)
                if isinstance(parsed, list):
                    vals = [normalize(strip_channel(x)) for x in parsed]
                    return ", ".join(dict.fromkeys(v for v in vals if v))
                if isinstance(parsed, dict):
                    if "amount" in parsed and "unit" in parsed:
                        return f"{parsed['amount']} {parsed['unit']}"
                    return ", ".join(f"{k}: {val}" for k, val in parsed.items())
            except json.JSONDecodeError:
                pass
        return normalize(strip_channel(t))
    return str(v)

def get_channel(value, channel="default"):
    """From a list of 'text[channel]' entries, return the one matching channel."""
    if isinstance(value, list):
        for item in value:
            m = re.search(r"\[([^\]]*)\]\s*$", str(item))
            if m and m.group(1) == channel:
                return strip_channel(item)
        if value:
            return strip_channel(value[0])
    return clean_value(value)


# ── Build one clean text blob per product ────────────────────────────────────
def product_to_text(identifier, attrs):
    """A product becomes a single, human-readable, embeddable document."""
    lines = []
    name = get_channel(attrs.get("Description 1", ""), "default") or identifier
    lines.append(f"Productnaam: {name}")
    lines.append(f"Artikelnummer: {identifier}")
    if attrs.get("GTIN"):
        lines.append(f"GTIN: {clean_value(attrs['GTIN'])}")
    for key in USEFUL_FIELDS + DESCRIPTION_FIELDS:
        if key in attrs:
            val = clean_value(attrs[key])
            if val:                      # skip empty fields entirely
                lines.append(f"{key}: {val}")
    return "\n".join(lines)


def load_products(path):
    """Accept a single .json file or a directory of them. One product each."""
    files = glob.glob(os.path.join(path, "*.json")) if os.path.isdir(path) else [path]
    products = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        identifier = data.get("identifier", os.path.basename(fp))
        attrs = data.get("attributes", {})
        products.append({
            "id": identifier,
            "text": product_to_text(identifier, attrs),
            "raw": attrs,                # keep originals as metadata for later
        })
    return products


# ── Cosine similarity (unchanged from earlier steps) ─────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    products = load_products(DATA_PATH)
    print(f"Loaded {len(products)} product(s).\n")

    # Show the clean blob for the first product so we can SEE what gets embedded.
    print("=" * 60)
    print("CLEAN TEXT BLOB (this is what we embed, one per product)")
    print("=" * 60)
    print(products[0]["text"])
    print("=" * 60, "\n")

    # Embed every product blob.
    print("Embedding products... ", end="", flush=True)
    for p in products:
        resp = ollama.embed(model=EMBED_MODEL, input=p["text"])
        p["vector"] = np.array(resp["embeddings"][0])
    print("done.\n")

    # Embed the query and rank products by similarity.
    q_resp = ollama.embed(model=EMBED_MODEL, input=QUERY)
    q_vec = np.array(q_resp["embeddings"][0])
    ranked = sorted(products,
                    key=lambda p: cosine_similarity(q_vec, p["vector"]),
                    reverse=True)
    top = ranked[:TOP_K]

    print(f'Query: "{QUERY}"\n')
    print(f"Top {len(top)} product(s) retrieved:")
    for i, p in enumerate(top, 1):
        score = cosine_similarity(q_vec, p["vector"])
        print(f"  {i}. {p['id']}  [score {score:.4f}]")
    print()

    # Build a grounded prompt from the retrieved product blobs.
    context = "\n\n".join(f"[Product {i+1}]\n{p['text']}" for i, p in enumerate(top))
    prompt = f"""Je bent een productadviseur. Beantwoord de vraag UITSLUITEND op
basis van de onderstaande productgegevens. Gebruik geen externe kennis. Als het
antwoord niet in de gegevens staat, zeg dan dat je het niet weet.

{context}

Vraag: {QUERY}""".strip()

    rag = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    print("ANSWER (with product context)")
    print("-" * 40)
    print(rag["message"]["content"], "\n")

    # Contrast: same question, no context. The model has never seen this product.
    bare = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": QUERY}])
    print("ANSWER (no context — bare question)")
    print("-" * 40)
    print(bare["message"]["content"])


if __name__ == "__main__":
    main()