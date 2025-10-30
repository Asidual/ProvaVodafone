# RAG Vodafone

Sistema RAG (Retrieval‑Augmented Generation) per manuali tecnici e FAQ, con pipeline di ingestion multimodale (testo + immagini), indicizzazione FAISS, API Backend (FastAPI) e Frontend separato.

---

## Contenuti

* [Panoramica](#panoramica)
* [Architettura](#architettura)
* [Struttura del repository](#struttura-del-repository)
* [Prerequisiti](#prerequisiti)
* [Configurazione `.env`](#configurazione-env)
* [Avvio rapido](#avvio-rapido)
* [Ingestion: manuali e immagini](#ingestion-manuali-e-immagini)
* [Ingestion: indice FAQ](#ingestion-indice-faq)
* [Vector DB (FAISS)](#vector-db-faiss)
* [API Backend](#api-backend)
* [Audit e logging](#audit-e-logging)
* [Sviluppo locale (senza Docker)](#sviluppo-locale-senza-docker)
* [Troubleshooting](#troubleshooting)

---

## Panoramica

**RAG Vodafone** indicizza contenuti di manualistica tecnica e relative figure estratte dal PDF. La risposta del modello viene “ancorata” alle evidenze recuperate dal Vector DB, con:

* Normalizzazione dei titoli markdown e split per capitolo/sezione
* Estrazione immagini con coordinate da Azure **Document Intelligence**
* Descrizione immagini via **VLM** (es. `gpt-4o`) e embedding testuale (OpenAI)
* Indicizzazione FAISS separata per: **contenuti del manuale** e **FAQ**
* API per `ask`, `fallback`, `warning` e modulo **audit**

---

## Architettura

```
PDF ──► Document Intelligence (markdown + figures json)
        │
        ├─► Normalizzazione headings
        ├─► Estrazione figure (PNG) + associazione a capitoli/sezioni
        ├─► Descrizioni immagini (VLM) ➜ embedding
        ├─► Testo dei blocchi ➜ embedding
        └─► Merge record (testo+immagini) + chunk ID
                         │
                         └─► FAISS (vectordb2)

FAQ.json ─► Validazione schema (Pydantic) ─► embedding (question)
                                      │
                                      └─► FAISS (FAQ)

FAISS + meta.json ─► Backend API (FastAPI) ─► FE
```

---

## Struttura del repository

```
BE/
├─ AllVectorDB/
│  ├─ vectordb2/           # index.faiss + meta.json del manuale
│  └─ FAQ/                 # FAQ.index + meta.json + index_info.json
│
├─ Ingestion/
│  ├─ Documents/           # sorgenti (PDF, json DI, md, debug_runs)
│  ├─ FAQ.py               # build indice FAISS per le FAQ [Sono state generate]
│  ├─ Ingestion.py         # Per ottenere il FAISS della documentazione del pdf
│  └─ utils.py             # funzioni DI/markdown/figures/embedding/FAISS
│  
├─ RagCode/
│  ├─ audit/               # audit.db, script ispezione
│  ├─ main.py              # entrypoint FastAPI
│  ├─ router.py            # endpoint API
│  └─ utils.py             # utilità backend
│
├─ Dockerfile              # backend
└─ Document/printf-manuale.pdf

FE/
├─ app.py                  # frontend
└─ Dockerfile

docker-compose.yml         # compose FE+BE
.env                        # variabili ambiente (non committare)
```

---

## Prerequisiti

* **Python 3.10+**
* **Docker** e **Docker Compose** (opzionale ma consigliato)
* Account e chiavi API:

  * **OpenAI** per embedding (es. `text-embedding-3-small/large`) e VLM (`gpt-4o`)
  * **Azure Document Intelligence** (endpoint + key) per conversione PDF→Markdown e coordinate immagini

---

## Configurazione `.env`

Crea un file `.env` nella root del progetto:

```ini
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
OPENAI_VLM_MODEL=gpt-4o

# Azure Document Intelligence (per pdf_to_markdown)
AZURE_DI_ENDPOINT=https://<nome-risorsa>.cognitiveservices.azure.com/
AZURE_DI_KEY=<chiave>
```

> Nota: la pipeline può anche partire da file già generati (markdown + json DI) senza chiamare DI a runtime.

---

## Avvio rapido

**Con Docker Compose (consigliato):**

```bash
docker-compose up --build
```

* BE in ascolto su `http://localhost:8000`
* FE in ascolto su `http://localhost:3000` (o porta configurata nel compose)

**Oppure solo Backend (uvicorn):**

```bash
# da root repo (con venv attivo e requirements installati)
python -m uvicorn BE.RagCode.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Ingestion: manuali e immagini

La pipeline completa con logging esteso è in `BE/Ingestion/pipeline_debug.py`.

**Percorsi di default usati nello script:**

* PDF: `BE/Ingestion/Documents/printf-manuale.pdf`
* Markdown: `BE/Ingestion/Documents/old_json/1_print_page4.md`
* Result DI json: `BE/Ingestion/Documents/old_json/1_result_info.json`
* Figure estratte: `BE\Ingestion\figures2`
* Output Vector DB: `BE/AllVectorDB/vectordb2/`

**Esecuzione:**

```bash
python BE/Ingestion/pipeline_debug.py
```

Crea/aggiorna `vectordb2` (FAISS + meta.json) e salva figure in `figures2/`.

Passi principali (funzioni in `utils.py`):

* `pdf_to_markdown(...)` (opzionale) – converte PDF→Markdown via Azure DI
* `normalize_numeric_headings`, `demote_unnumbered_headers_to_bold` – normalizza heading
* `extract_figures_png_robust_from_dict` – estrae PNG con bounding box DI
* `split_markdown_build_records_by_page_markers_no_recursive` – crea record per blocchi
* `correct_fig_assigment` – fix manuale dell’associazione immagine→sezione
* `add_embedding` / `add_embedding_image` – embedding testo/descrizioni (OpenAI)
* `merge_text_and_image_records` + `add_chunk_ids` – unificazione + ID stabili
* `build_vectordb` – scrive `index.faiss` + `meta.json`

---

## Ingestion: indice FAQ

`BE/Ingestion/FAQ.py` costruisce un indice FAISS dedicato alle FAQ, embeddando **solo la question** e salvando i metadati (question/answer/id).

**Input atteso:** `BE/Ingestion/Documents/FAQs/FAQ.json` (lista di oggetti con `{id, question, answer, source?}`)

**Esecuzione:**

```bash
python BE/Ingestion/FAQ.py
```

Scrive in `BE/AllVectorDB/FAQ/`:

* `FAQ.index` (FAISS)
* `meta.json` (mappa id → {question, answer})
* `index_info.json` (modello, dim, count, timestamp)

Parametri principali:

* `EMBED_MODEL = "text-embedding-3-large"` (1536‑dim)
* Similarità: **cosine** (vettori normalizzati; IndexFlatIP)

---

## Vector DB (FAISS)

* **Manuale**: `BE/AllVectorDB/vectordb2/`

  * `index.faiss` – vettori normalizzati (cosine via inner product)
  * `meta.json` – lista completa dei chunk (testo/immagini + metadati)
* **FAQ**: `BE/AllVectorDB/FAQ/`

  * `FAQ.index`, `meta.json`, `index_info.json`

Utility di ricerca disponibile in `utils.py`:

```python
from BE.Ingestion.utils import search_by_vector

results = search_by_vector(
    db_dir="BE/AllVectorDB/vectordb2",
    query_vector=<embedding della query>,
    top_k=5,
)
```

---

## API Backend

Entrypoint FastAPI in `BE/RagCode/main.py` e router in `BE/RagCode/router.py`.

Endpoint principali (nomenclature indicative):

* `POST /ask` o `/ask/stream` – risposta con contesto da Vector DB
* `POST /fallback` – proposte di FAQ correlate quando il match è scarso
* `POST /warning` – validazioni/flag tecnici su domanda/risposta

> Vedi `router.py` e i docstring per i parametri esatti. Il BE salva eventi su **audit.db**.

Esempio (curl generico):

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Come cambio il rotolo?", "top_k": 5}'
```

---

## Audit e logging

* DB: `BE/RagCode/audit/audit.db`
* Script utili: `audit_inspect.py`, `audit.py`
* Log pipeline: `pipeline_debug.py` → livello `DEBUG` (console) + salvataggi intermedi in `BE/Ingestion/Documents/debug_runs/`

---

## Sviluppo locale (senza Docker)

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 1) Costruisci indice FAQ (opzionale)
python BE/Ingestion/FAQ.py

# 2) Esegui ingestion del manuale
python BE/Ingestion/pipeline_debug.py

# 3) Avvia Backend
python -m uvicorn BE.RagCode.main:app --reload --port 8000
```

---

## Troubleshooting

* **`OPENAI_API_KEY non impostata`** → verifica `.env` e variabili d’ambiente
* **Mancano file DI (markdown/json)** → lancia `pdf_to_markdown` o allinea i percorsi in `pipeline_debug.py`
* **Immagini non trovate** in `describe_figure` → controlla path relativi/assoluti e cartella `figures2/`
* **Dimensioni embedding** → `text-embedding-3-small` (1536‑d). Se cambi modello, ricrea l’indice.
* **FAISS non trovato** → su alcune piattaforme serve `faiss-cpu` compatibile con la tua Python/OS

---

## Licenza

Proprietario. Uso interno a progetto Vodafone.
