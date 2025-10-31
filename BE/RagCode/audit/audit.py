# BE/RagCode/audit/audit.py
from __future__ import annotations
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =========================
# Path DB (funziona sia in locale che in Docker)
# - Default: /app/RagCode/audit/audit.db (grazie al volume in docker-compose)
# - Override con env AUDIT_DB se vuoi
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]  # -> .../RagCode
DB_PATH: Path = Path(
    os.getenv("AUDIT_DB", str(BASE_DIR / "audit" / "audit.db"))
).resolve()

# =========================
# Schema + PRAGMA (WAL per scritture concorrenti; sync NORMAL)
# =========================
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS audit_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc INTEGER NOT NULL,
  route TEXT NOT NULL,
  user_id TEXT,
  question_id TEXT,
  question TEXT,
  answer_status TEXT,     -- "OK" | "KO" | "PARTIAL" | NULL
  model TEXT,
  temperature REAL,
  max_tokens INTEGER,
  latency_ms INTEGER,
  meta_json TEXT
);

CREATE TABLE IF NOT EXISTS audit_resource (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  audit_id INTEGER NOT NULL,
  doc_id TEXT,
  doc_title TEXT,
  doc_version TEXT,
  page INTEGER,
  score REAL,
  chunk_id TEXT,
  FOREIGN KEY(audit_id) REFERENCES audit_event(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audit_event_ts   ON audit_event(ts_utc);
CREATE INDEX IF NOT EXISTS idx_audit_event_qid  ON audit_event(question_id);
CREATE INDEX IF NOT EXISTS idx_audit_res_aid    ON audit_resource(audit_id);
"""

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def _has_column(cx: sqlite3.Connection, table: str, col: str) -> bool:
    cur = cx.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def init_db() -> None:
    """
    Inizializza/migra il DB: crea tabelle/indici (idempotente)
    e aggiunge eventuali colonne mancanti.
    """
    with _connect() as cx:
        for stmt in [s.strip() for s in _SCHEMA.strip().split(";") if s.strip()]:
            cx.execute(stmt + ";")

        # Soft-migration di sicurezza: assicurati che question_id esista
        if not _has_column(cx, "audit_event", "question_id"):
            cx.execute("ALTER TABLE audit_event ADD COLUMN question_id TEXT;")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_qid ON audit_event(question_id);")

def start_event(
    route: str,
    question: Optional[str],
    *,
    user_id: Optional[str] = None,
    question_id: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Crea una riga su audit_event e ritorna audit_id.
    """
    with _connect() as cx:
        cur = cx.cursor()
        cur.execute(
            """INSERT INTO audit_event
               (ts_utc, route, user_id, question_id, question, answer_status, model, temperature, max_tokens, latency_ms, meta_json)
               VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?)""",
            (
                int(time.time()),
                route,
                user_id,
                question_id,
                (question or "")[:8000],  # limite prudenziale
                model or "",
                float(temperature) if temperature is not None else None,
                int(max_tokens) if max_tokens is not None else None,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)

def add_resources(audit_id: int, resources: Iterable[Dict[str, Any]]) -> None:
    """
    Inserisce righe su audit_resource.
    Ogni dict può contenere: id/doc_id, title/doc_title, version/doc_version, page, score, chunk_id.
    """
    rows= []
    for r in resources or []:
        rows.append((
            audit_id,
            r.get("id") or r.get("doc_id"),
            r.get("title") or r.get("doc_title"),
            r.get("version") or r.get("doc_version"),
            r.get("page"),
            float(r.get("score")) if r.get("score") is not None else None,
            r.get("chunk_id"),
        ))
    if not rows:
        return
    with _connect() as cx:
        cx.executemany(
            """INSERT INTO audit_resource
               (audit_id, doc_id, doc_title, doc_version, page, score, chunk_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

def finish_event(
    audit_id: int,
    *,
    answer_status: Optional[str] = None,   # "OK" | "KO" | "PARTIAL" | ...
    latency_ms: Optional[int] = None,
    meta_update: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Aggiorna stato finale/latency e merge di meta_json con meta_update.
    """
    with _connect() as cx:
        if meta_update:
            cur = cx.execute("SELECT meta_json FROM audit_event WHERE id=?", (audit_id,))
            row = cur.fetchone()
            base = {}
            if row and row[0]:
                try:
                    base = json.loads(row[0])
                except Exception:
                    base = {}
            base.update(meta_update)
            cx.execute(
                "UPDATE audit_event SET meta_json=? WHERE id=?",
                (json.dumps(base, ensure_ascii=False), audit_id),
            )

        cx.execute(
            """UPDATE audit_event
               SET answer_status=COALESCE(?, answer_status),
                   latency_ms=COALESCE(?, latency_ms)
               WHERE id=?""",
            (answer_status, latency_ms, audit_id),
        )
