"""
RAG pipeline with a persistent ChromaDB vector store.

What changed from rag_pdf.py / the numpy-only version:
  - Vectors are saved to ./chroma_db on disk.
  - On startup we only embed products that are NOT already in the collection.
  - Everything else (JSON loading, field curation, ollama embedding, llama3.2
    generation) is identical to the previous step.

WHERE EMBEDDING HAPPENS
  We call ollama.embed() ourselves and hand the resulting numpy array to Chroma.
  Chroma never touches Ollama. This is called "bring your own embeddings" (BYOE)
  and is the safest approach when you want full control over the embedding model.
"""

import os
import re
import json
import glob
import time
import numpy as np
import ollama
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "nomic-embed-text-v2-moe"
CHAT_MODEL   = "llama3.2"
TOP_K        = 8
DATA_PATH    = r"C:\Users\marbos\Downloads\selected_products"
CHROMA_DIR   = "./chroma_db"        # folder Chroma creates / reads on disk
COLLECTION   = "cotap_products"     # logical name; Chroma supports many collections
QUERY        = "Welke vloer is waterbestendig en geschikt voor de badkamer?"

# ── Field curation (unchanged from previous step) ─────────────────────────────
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
DESCRIPTION_FIELDS = ["Cotap artikeltekst", "Brandsites artikeltekst"]

# ── Value cleaning (unchanged) ────────────────────────────────────────────────
def strip_channel(s):
    return re.sub(r"\[[^\]]*\]\s*$", "", str(s)).strip()

def normalize(s):
    s = str(s).replace("_x002D_", "-")
    return s.replace("__", " ").replace("_", " ").strip()

def clean_value(v):
    if isinstance(v, list):
        seen = []
        for x in v:
            t = normalize(strip_channel(x))
            if t and t not in seen:
                seen.append(t)
        return ", ".join(seen)
    if isinstance(v, str):
        t = v.strip()
        if t.startswith("[") or t.startswith("{"):
            try:
                parsed = json.loads(t)
                if isinstance(parsed, list):
                    return ", ".join(dict.fromkeys(
                        normalize(strip_channel(x)) for x in parsed
                        if normalize(strip_channel(x))
                    ))
                if isinstance(parsed, dict):
                    if "amount" in parsed and "unit" in parsed:
                        return f"{parsed['amount']} {parsed['unit']}"
                    return ", ".join(f"{k}: {val}" for k, val in parsed.items())
            except json.JSONDecodeError:
                pass
        return normalize(strip_channel(t))
    return str(v)

def get_channel(value, channel="default"):
    if isinstance(value, list):
        for item in value:
            m = re.search(r"\[([^\]]*)\]\s*$", str(item))
            if m and m.group(1) == channel:
                return strip_channel(item)
        if value:
            return strip_channel(value[0])
    return clean_value(value)

def product_to_text(identifier, attrs):
    lines = []
    name = get_channel(attrs.get("Description 1", ""), "default") or identifier
    lines.append(f"Productnaam: {name}")
    lines.append(f"Artikelnummer: {identifier}")
    if attrs.get("GTIN"):
        lines.append(f"GTIN: {clean_value(attrs['GTIN'])}")
    for key in USEFUL_FIELDS + DESCRIPTION_FIELDS:
        if key in attrs:
            val = clean_value(attrs[key])
            if val:
                lines.append(f"{key}: {val}")
    return "\n".join(lines)

def load_products(path):
    files = glob.glob(os.path.join(path, "*.json")) if os.path.isdir(path) else [path]
    products = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        identifier = data.get("identifier", os.path.basename(fp))
        attrs = data.get("attributes", {})
        products.append({
            "id":   identifier,
            "text": product_to_text(identifier, attrs),
            "raw":  attrs,
        })
    return products

# ── NEW: extract structured metadata for each product ────────────────────────
# Chroma stores metadata alongside each vector. We don't USE it for filtering
# yet, but storing it now means we can add WHERE clauses later with zero
# re-embedding. Chroma requires metadata values to be str/int/float/bool —
# never None — so we fall back to "" for missing fields.
def extract_metadata(identifier, attrs):
    def safe(key):
        val = clean_value(attrs[key]) if key in attrs else ""
        return val if val else ""   # Chroma rejects None

    return {
        "product_type":      safe("Product type"),
        "categorie_1":       safe("Categorie 1"),
        "kleurfamilie":      safe("Kleurfamilie"),
        "trapgeschiktheid":  safe("Trapgeschiktheid"),
        # "name" is handy to have in metadata so we can print it from query results
        "name": get_channel(attrs.get("Description 1", ""), "default") or identifier,
    }

# ── NEW: set up the persistent Chroma collection ─────────────────────────────
def get_collection():
    # PersistentClient writes to disk at CHROMA_DIR. No server process needed.
    # On first run: creates the folder and an empty collection.
    # On later runs: opens the existing collection — vectors already there.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # get_or_create_collection is idempotent: safe to call every run.
    # space="cosine" tells Chroma to use cosine distance so results are
    # comparable to the cosine SIMILARITY we computed by hand before.
    # ⚠  IMPORTANT — Chroma returns DISTANCE, not similarity.
    #    distance = 1 − cosine_similarity
    #    So distance 0.0 = identical, distance 1.0 = unrelated.
    #    This is the INVERSE of what our old script printed (where 1.0 = best).
    #    That's expected — lower Chroma distance means better match.
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

# ── NEW: sync — embed only products missing from the collection ───────────────
def sync_products(collection, products):
    all_ids = [p["id"] for p in products]

    # Ask Chroma which of these IDs it already knows.
    # include=[] means "return IDs only, skip vectors/documents" — fast.
    existing = collection.get(ids=all_ids, include=[])
    existing_ids = set(existing["ids"])

    new_products = [p for p in products if p["id"] not in existing_ids]
    cached_count = len(existing_ids)
    new_count    = len(new_products)

    print(f"  {cached_count} product(s) already in collection → skipping embedding.")
    print(f"  {new_count} new product(s) → embedding now…")

    if not new_products:
        return   # nothing to do

    # Embed and add in one batch call to keep things readable.
    # We pass embeddings= ourselves — Chroma stores them as-is.
    ids, embeddings, documents, metadatas = [], [], [], []
    for i, p in enumerate(new_products, 1):
        resp  = ollama.embed(model=EMBED_MODEL, input=p["text"])
        vec   = resp["embeddings"][0]   # plain Python list; Chroma is fine with that
        ids.append(p["id"])
        embeddings.append(vec)
        documents.append(p["text"])
        metadatas.append(extract_metadata(p["id"], p["raw"]))
        print(f"    embedded {i}/{new_count}: {p['id']}", end="\r", flush=True)

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"\n  Done. Collection now holds {collection.count()} product(s).")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Phase 1: load JSON from disk + sync Chroma cache ─────────────────────
    t0 = time.perf_counter()

    products   = load_products(DATA_PATH)
    collection = get_collection()

    print(f"\nLoaded {len(products)} product(s) from disk.")
    print(f"Syncing with Chroma collection '{COLLECTION}' at {CHROMA_DIR}/ …\n")
    sync_products(collection, products)

    t1 = time.perf_counter()

    # ── Phase 2: embed the query ──────────────────────────────────────────────
    # We embed the query ourselves (same model, same pattern as before).
    q_resp = ollama.embed(model=EMBED_MODEL, input=QUERY)
    q_vec  = q_resp["embeddings"][0]

    t2 = time.perf_counter()

    # ── Phase 3: Chroma retrieval ─────────────────────────────────────────────
    # collection.query() returns a dict; each value is a list-of-lists because
    # Chroma supports batched queries.  We sent one query, so we index with [0].
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # -- Commented example of a METADATA FILTER (not active yet) ---------------
    # To restrict results to floor products that are stair-suitable, add:
    #   where={"trapgeschiktheid": {"$ne": ""}}
    # or a more specific filter:
    #   where={"$and": [{"categorie_1": "Tapijt"}, {"trapgeschiktheid": "trapgeschikt"}]}
    # ChromaDB supports $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or.
    # --------------------------------------------------------------------------

    ids        = results["ids"][0]
    distances  = results["distances"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]

    t3 = time.perf_counter()

    print(f'\nQuery: "{QUERY}"\n')
    print(f"Top {len(ids)} match(es)  [distance: lower = more similar]:\n")
    for i, (pid, dist, meta) in enumerate(zip(ids, distances, metadatas), 1):
        # Convert Chroma cosine distance back to a similarity for readability:
        #   similarity = 1 − distance  (only valid with hnsw:space = cosine)
        similarity = 1.0 - dist
        print(f"  {i}. {pid}  |  dist {dist:.4f}  (similarity {similarity:.4f})")
        print(f"     {meta.get('name', '–')}")
    print()

    # ── Phase 4: grounded, cited generation ──────────────────────────────────
    # Context passages are in the order Chroma returned them (most similar first).
    # We label them [Product 1], [Product 2], … and tell the model that lower
    # numbers are more relevant, so it can weight them accordingly.
    context = "\n\n".join(
        f"[Product {i+1}]\n{doc}" for i, doc in enumerate(documents)
    )

    # WHY this prompt structure:
    #   - "UITSLUITEND op basis van" (only from) prevents the model drawing on
    #     its training data, which would make claims unverifiable.
    #   - The citation rule "(bron: …)" ties every claim to a specific field in a
    #     specific product, so a user can open the JSON and check it themselves.
    #   - The explicit fallback sentence stops the model from confabulating a
    #     plausible-sounding but wrong answer when no product actually fits.
    #   - Telling the model which passage is most relevant reduces the "lost in
    #     the middle" effect: LLMs tend to ignore context that appears in the
    #     middle of a long prompt unless told it matters.
    prompt = f"""Je bent een productadviseur voor een vloerenbedrijf.

Instructies — lees deze zorgvuldig voordat je antwoordt:
1. Beantwoord de vraag UITSLUITEND op basis van de onderstaande productpassages.
   Gebruik geen externe kennis of aannames buiten de gegeven tekst.
2. Onderbouw elke bewering met een bronvermelding: noem de Productnaam, het
   Artikelnummer en het specifieke veld waarop je je baseert.
   Gebruik dit formaat: (bron: <Productnaam>, art. <Artikelnummer>, veld '<Veldnaam>: <waarde>')
   Voorbeeld: (bron: Expression Aqua zand, art. EXP-001, veld 'Geschikt voor (ruimte): badkamer')
3. De passages zijn gerangschikt op relevantie: [Product 1] is het meest relevant
   voor de vraag, [Product {TOP_K}] het minst. Geef voorrang aan vroegere passages.
4. Als GEEN van de producten duidelijk voldoet aan de vraag, antwoord dan
   letterlijk met: "Geen van de gevonden producten voldoet duidelijk aan deze vraag."
   Doe geen gissingen en verzin geen producteigenschappen.

Productpassages:

{context}

Vraag: {QUERY}""".strip()

    # CHANGE 1 — temperature=0: every token is chosen deterministically (the
    # highest-probability token is always picked, no randomness).  The same
    # prompt will now produce the same answer every run, which makes the output
    # testable and comparable across query changes.
    rag = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    t4 = time.perf_counter()

    print("ANSWER (with product context)")
    print("-" * 40)
    print(rag["message"]["content"])

    # ── Timing summary ────────────────────────────────────────────────────────
    # perf_counter() is the highest-resolution clock available in Python.
    # We measure wall-clock time, so network latency to the local Ollama daemon
    # is included in the embed and generate phases — that's intentional, since
    # it reflects the real cost of each step.
    print()
    print("── Timing breakdown ─────────────────────────────────")
    print(f"  Load + sync   : {t1 - t0:6.2f} s")
    print(f"  Embed query   : {t2 - t1:6.2f} s")
    print(f"  Chroma lookup : {t3 - t2:6.2f} s")
    print(f"  Generate      : {t4 - t3:6.2f} s")
    print(f"  ─────────────────────────────")
    print(f"  Total         : {t4 - t0:6.2f} s")

if __name__ == "__main__":
    main()
