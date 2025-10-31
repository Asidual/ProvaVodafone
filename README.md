
# RAG Vodafone

Sistema **RAG (Retrieval-Augmented Generation)** per consultare **manuali tecnici e FAQ**, con:

* ingestion multimodale (testo + immagini)
* indicizzazione **FAISS**
* **Backend** FastAPI (con SSE per streaming token)
* **Frontend** Streamlit
* **Audit** locale su SQLite

---

## Struttura del repository

```
BE/
├─ AllVectorDB/
│  ├─ FAQ/                     # indice FAISS delle FAQ + metadati
│  └─ vectordb2/               # indice FAISS del manuale + metadati
│
├─ Ingestion/
│  ├─ Documents/               # PDF, json/markdown DI, debug
│  │  ├─ debug_runs/           # Ci sono tutti i documenti tra Json e MD per i passaggi intermedi per la fase di ingestion
│  │  ├─ FAQs/                 # FAQ.json (input)
│  │  └─ old_json/             # output Azure Document Intelligence (facoltativo)
│  ├─ figures2/                # immagini PNG estratte
│  ├─ FAQ.py                   # builder indice FAQ (FAISS)
│  ├─ ingestion.py             # builder indice manuale (FAISS)
│  └─ utils.py                 # funzioni supporto ingestion
│
├─ RagCode/
│  ├─ audit/                   # audit.db + helper (salvato anche come volume)
│  ├─ Document/                # pdf manuale (servito come /staticdoc)
│  ├─ main.py                  # app FastAPI (mount static, VDB loader)
│  ├─ router.py                # API: /search, /ask, /ask/stream, /fallback, /warning
│  └─ utils.py                 # utilità backend (embed, ricerca, prompt, ecc.)
│
├─ Dockerfile                  # backend
└─ requirements.txt            # backend

FE/
├─ app.py                      # Streamlit app
├─ Dockerfile                  # frontend
└─ requirements.txt            # frontend

docker-compose.yml
.env
command_docker.md
README.md
```

E' presente anche una cartella chiamata **Notebook** che ho utilizzato per gli esperimenti e per le varie prove nella fase di ingestion

---

## Variabili d’ambiente (.env)

Crea un file **`.env`** nella root (stesso livello del `docker-compose.yml`):

```ini
# ==== OPENAI ====
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-large

# ==== BACKEND ====
APP_NAME=RAG BE
APP_VERSION=0.2.0
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
RELOAD=false

# Windows-safe: accetta anche backslash; vengono normalizzati in runtime
VDB_DIR=BE/AllVectorDB/vectordb2
STATIC_DIR=BE/Ingestion/figures2
STATIC_DIR_DOC=BE/RagCode/Document

# Audit
AUDIT_ENABLED=true

# ==== FRONTEND ====
# lato container FE -> come il FE chiama il BE (via rete docker)
API_BASE_URL=http://backend:8000/api

# lato browser -> come l'utente vede il BE (fuori da docker)
PUBLIC_API_BASE_URL=http://localhost:8000/api
```

> `API_BASE_URL` è usato **dal container FE** per chiamare il BE (`backend:8000`).
> `PUBLIC_API_BASE_URL` è usato **dal browser** per aprire link alle immagini/PDF (`http://localhost:8000/...`).
> Se non coincidono, le immagini non si aprono dal browser.

---

## Avvio con Docker (consigliato)

```bash
docker compose up --build
```

* **Backend** → [http://localhost:8000](http://localhost:8000)

  * docs: [http://localhost:8000/docs](http://localhost:8000/docs)
  * static immagini: [http://localhost:8000/static/](http://localhost:8000/static/)<file.png>
  * static documenti: [http://localhost:8000/staticdoc/printf-manuale.pdf](http://localhost:8000/staticdoc/printf-manuale.pdf)
  * debug immagini: [http://localhost:8000/debug/static-list](http://localhost:8000/debug/static-list)
* **Frontend** → [http://localhost:8501](http://localhost:8501)

Comandi utili:

```bash
# ricostruzione completa dopo modifiche a requirements/Dockerfile
docker compose build --no-cache

# log live di un servizio
docker compose logs -f backend
docker compose logs -f frontend

# riavvio "pulito"
docker compose down
docker compose up -d --build
```

### Volumi mappati (persistenza e hot-reload)

Nel `docker-compose.yml`:

* `./BE:/app` → codice BE “montato” (modifiche visibili senza rebuild)
* `./BE/RagCode/audit:/app/RagCode/audit` → **audit.db** persistente
* `./BE/AllVectorDB:/app/AllVectorDB` → indici FAISS persistenti
* `./BE/RagCode/.logs:/app/RagCode/.logs` → log su host

---

## Avvio locale (senza Docker)

Backend:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r BE/requirements.txt
python -m uvicorn BE.RagCode.main:app --reload --port 8000
```

Frontend:

```bash
pip install -r FE/requirements.txt
# Sovrascrivi per l’avvio locale:
set API_BASE_URL=http://localhost:8000/api
set PUBLIC_API_BASE_URL=http://localhost:8000/api
streamlit run FE/app.py --server.port=8501
```

---

## Pipeline di ingestion

### Manuale + immagini → `vectordb2/`

```bash
python BE/Ingestion/ingestion.py
```

Output atteso:

```
BE/AllVectorDB/vectordb2/
├─ index.faiss
└─ meta.json
```

### FAQ → `FAQ/`

Assicurati di avere `BE/Ingestion/Documents/FAQs/FAQ.json` (lista di oggetti con `{id, question, answer, ...}`):

```bash
python BE/Ingestion/FAQ.py
```

Output atteso:

```
BE/AllVectorDB/FAQ/
├─ FAQ.index
├─ meta.json
└─ index_info.json
```

---

## API principali (router FastAPI)

| Metodo | Endpoint          | Descrizione                                                                         |
| -----: | ----------------- | ------------------------------------------------------------------------------------|
|   POST | `/api/search`     | Ricerca semantica nei chunk FAISS e spiegabilità dei documenti selezionati          |
|   POST | `/api/ask`        | Generazione risposta (sincrona)                                                     |
|   POST | `/api/ask/stream` | Generazione **streaming** (SSE token-by-token)                                      |
|   POST | `/api/fallback`   | FAQ correlate se retrieval scarso                                                   |
|   POST | `/api/warning`    | Analisi e flag di warning tecnico                                                   |

Endpoint utilità:

* `GET /` → stato app + config statici
* `GET /debug/static-list` → anteprima file immagini montati

> **SSE (Server-Sent Events)**: nel backend usi `StreamingResponse(..., media_type="text/event-stream")` e nel FE leggi riga per riga con `requests.iter_lines()`.

---

## Audit & Logging

* DB: `BE/RagCode/audit/audit.db` (montato come volume)
* Tabelle: `audit_event`, `audit_resource`
* Log file: `BE/RagCode/.logs/app.log` (+ `rag_stream.log` dal router)

Ispezione rapida (se hai `sqlite3`):

```bash
sqlite3 BE/RagCode/audit/audit.db ".tables"
sqlite3 BE/RagCode/audit/audit.db "SELECT id, ts_utc, route, question_id, answer_status, latency_ms FROM audit_event ORDER BY id DESC LIMIT 10;"
```

---

## Settaggi consigliati

**Backend (.env)**

* `HOST=0.0.0.0` (in Docker) / `HOST=127.0.0.1` (locale)
* `RELOAD=false` in produzione
* `LOG_LEVEL=INFO` (passa a `DEBUG` solo in dev)
* `VDB_DIR=BE/AllVectorDB/vectordb2`
* `STATIC_DIR=BE/Ingestion/figures2`
* `STATIC_DIR_DOC=BE/RagCode/Document`
* `AUDIT_ENABLED=true`

**Frontend (.env)**

* `API_BASE_URL=http://backend:8000/api` (FE → BE via rete Docker)
* `PUBLIC_API_BASE_URL=http://localhost:8000/api` (browser → BE)

**Modelli**

* `OPENAI_CHAT_MODEL=gpt-4o` (veloce/economico)
* `OPENAI_EMBED_MODEL=text-embedding-3-large` (1536-d, qualità buona)
* FE default: `Top-K = 5`, `temperature = 0.2`, `max_tokens = 700`

---

## Check rapido (smoke test)

1. BE su: [http://localhost:8000](http://localhost:8000)

   * verifica `/`, `/docs`, `/debug/static-list`

2. Immagini: prendi un file elencato da `/debug/static-list`

   * esempio: `http://localhost:8000/static/page7_figure4.png`

3. FE su: [http://localhost:8501](http://localhost:8501)

   * in **sidebar** verifica:

     ```
     API_BASE_URL = http://backend:8000/api
     PUBLIC_API_BASE_URL = http://localhost:8000/api
     ```
   * fai una query: dovresti vedere i link alle fonti e le immagini aprirsi nel browser.

---

## Troubleshooting

| Problema                                                          | Possibile causa                                   | Soluzione                                                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Le immagini aprono su `backend:8000/...` e il browser non risolve | `PUBLIC_API_BASE_URL` non impostata correttamente | Metti `PUBLIC_API_BASE_URL=http://localhost:8000/api` (o l’hostname pubblico)                           |
| 404 o vuoto su `/static/...`                                      | Cartella figure non montata o path errato         | Verifica `STATIC_DIR` e `docker compose logs -f backend` → vedi “Mount statico immagini: …”             |
| FE dice “Connection refused” su `/api/search`                     | Backend giù o porta non esposta                   | `docker compose ps`, controlla che `8000:8000` sia in LISTEN, guarda i log                              |
| Audit non scrive                                                  | Volume non montato o permessi                     | Verifica mapping `./BE/RagCode/audit:/app/RagCode/audit` e che il container possa scrivere              |
| `ModuleNotFoundError: BE` in locale                               | pacchetto non inizializzabile                     | Assicurati che ci siano `__init__.py` (ci sono già) e lanci con `python -m uvicorn BE.RagCode.main:app` |
| VDB “0 chunk”                                                     | Indice non generato                               | Esegui `BE/Ingestion/ingestion.py` (manuale) e/o `FAQ.py`                                               |

---

