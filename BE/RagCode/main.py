# BE/RagCode/main.py
import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from router import rag_router
from utils import load_vectordb

# --- AUDIT ---
from audit.audit import init_db, DB_PATH  # <- integrazione audit

load_dotenv()

# =========================
# Path base e normalizzazione
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

def _normalize_path(p: str | None, *fallback: str) -> Path:
    """
    - Accetta path da env con backslash di Windows.
    - Se non presente, usa fallback relativo a BASE_DIR.
    - Restituisce Path assoluto.
    - Utile per Docker
    """
    if p and p.strip():
        p = p.replace("\\", "/")
        path = Path(p)
        # Se è relativo, ancoralo a BASE_DIR
        if not path.is_absolute():
            path = BASE_DIR / path
    else:
        path = BASE_DIR.joinpath(*fallback)
    return path.resolve()


# =========================
# Config
# =========================
APP_NAME  = os.getenv("APP_NAME", "RAG BE")
APP_VER   = os.getenv("APP_VERSION", "0.2.0")

# Auto-switch HOST: locale -> 127.0.0.1, Docker -> 0.0.0.0 (se non forzato da .env)
HOST = os.getenv("HOST", "localhost")

PORT      = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = os.getenv("LOG_FILE", "").strip()
RELOAD    = os.getenv("RELOAD", "false").lower() == "true"

# Audit toggle
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() == "true"

# Vector DB
VDB_DIR   = _normalize_path(os.getenv("VDB_DIR"), "AllVectorDB", "vectordb2")

# Static (immagini)
STATIC_MOUNT     = (os.getenv("STATIC_MOUNT", "/static") or "/static").rstrip("/") or "/static"
STATIC_BASE_URL  = "http://localhost:8000/api"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
STATIC_DIR       = _normalize_path(os.getenv("STATIC_DIR"), "Ingestion", "figures2")

# Static (documenti)
STATIC_MOUNT_DOC = "/staticdoc"
STATIC_DIR_DOC   = _normalize_path(os.getenv("STATIC_DIR_DOC"), "RagCode", "Document")


# =========================
# Logging
# =========================
def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=LOG_LEVEL, format=fmt, datefmt=datefmt)

    logger = logging.getLogger(APP_NAME.replace(" ", "_"))

    if LOG_FILE:
        try:
            lf = Path(LOG_FILE.replace("\\", "/"))
            if not lf.is_absolute():
                lf = BASE_DIR / lf
            lf.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(str(lf), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
            fh.setLevel(LOG_LEVEL)
            fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logger.addHandler(fh)
            logger.info("Logging su file: %s", lf)  # <- piccola conferma utile
        except Exception:  # non bloccare l'avvio se fallisce il file log
            logger.exception("Impossibile configurare il file di log: %s", LOG_FILE)

    # riduci rumore
    logging.getLogger("uvicorn.error").setLevel(LOG_LEVEL)
    logging.getLogger("uvicorn.access").setLevel(LOG_LEVEL)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    return logger

logger = setup_logging()


# =========================
# Lifespan (audit + caricamento VDB)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Audit DB ---
    if AUDIT_ENABLED:
        try:
            init_db()
            logger.info("Audit DB inizializzato: %s", str(DB_PATH))
        except Exception:
            logger.exception("Errore inizializzazione Audit DB")

    # --- Vector DB ---
    logger.info("Avvio… caricamento Vector DB: %s", VDB_DIR)
    try:
        index, meta = load_vectordb(str(VDB_DIR))
        app.state.vdb_index = index
        app.state.vdb_meta  = meta
        app.state.vdb_dir   = str(VDB_DIR)
        logger.info("Vector DB caricato: %d chunk", len(meta))
    except Exception:
        logger.exception("Errore nel caricamento del Vector DB")
        app.state.vdb_index = None
        app.state.vdb_meta  = []
        app.state.vdb_dir   = str(VDB_DIR)

    yield
    logger.info("Arresto applicazione…")


# =========================
# App
# =========================
app = FastAPI(title=APP_NAME, version=APP_VER, lifespan=lifespan)

# ---- Static: immagini ----
if STATIC_DIR.is_dir():
    logger.info("Mount statico immagini: %s -> %s", STATIC_MOUNT, STATIC_DIR)
    app.mount(STATIC_MOUNT, StaticFiles(directory=str(STATIC_DIR)), name="static")
    try:
        files = [f for f in os.listdir(STATIC_DIR)]
        sample = f"{STATIC_MOUNT}/{files[0]}" if files else None
        logger.info("Esempio URL statico: %s", sample)
    except Exception:
        logger.exception("Impossibile elencare la cartella immagini: %s", STATIC_DIR)
else:
    logger.warning("STATIC_DIR non trovato: %s (skip mount)", STATIC_DIR)

# ---- Static: documenti ----
if STATIC_DIR_DOC.is_dir():
    logger.info("Mount statico documenti: %s -> %s", STATIC_MOUNT_DOC, STATIC_DIR_DOC)
    app.mount(STATIC_MOUNT_DOC, StaticFiles(directory=str(STATIC_DIR_DOC)), name="static_doc")
else:
    logger.warning("STATIC_DIR_DOC non trovato: %s (skip mount)", STATIC_DIR_DOC)


# ---- Debug: lista file statici con URL pubblici
@app.get("/debug/static-list")
def static_list():
    if not STATIC_DIR.is_dir():
        return {"dir_exists": False, "dir": str(STATIC_DIR), "files": []}

    try:
        files = sorted([
            f for f in os.listdir(STATIC_DIR)
        ])
    except Exception:
        logger.exception("Errore lettura cartella static")
        return {"dir_exists": True, "dir": str(STATIC_DIR), "count": 0, "files": [], "urls": []}

    urls_rel = [f"{STATIC_MOUNT}/{f}" for f in files[:10]]
    urls_full = [f"{PUBLIC_BASE_URL}{u}" if PUBLIC_BASE_URL else u for u in urls_rel]

    return {
        "dir_exists": True,
        "dir": str(STATIC_DIR),
        "count": len(files),
        "files": files[:50],
        "urls": urls_full,
        "urls_rel": urls_rel,
    }



# ---- Router API ----
app.include_router(rag_router, prefix="/api", tags=["rag"])


@app.get("/")
def root():
    return {
        "status": "ok",
        "ready": app.state.vdb_index is not None,
        "chunks": len(getattr(app.state, "vdb_meta", [])),
        "vdb_dir": getattr(app.state, "vdb_dir", None),
        "static": {
            "mount": STATIC_MOUNT,
            "dir": str(STATIC_DIR),
            "doc_mount": STATIC_MOUNT_DOC,
            "doc_dir": str(STATIC_DIR_DOC),
            "public_base_url": PUBLIC_BASE_URL or None,
        },
        "audit": {
            "enabled": AUDIT_ENABLED,
            "db_path": str(DB_PATH),
        },
    }


# =========================
# Run locale
# =========================
if __name__ == "__main__":
    logger.info("Uvicorn su %s:%d (reload=%s)", HOST, PORT, RELOAD)
    uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD,
                log_level=LOG_LEVEL.lower(), access_log=True)
