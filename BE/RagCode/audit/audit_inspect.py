# audit/audit_inspect.py — CLI per ispezionare l'audit log (SQLite o JSONL)
import argparse, csv, json, sqlite3, sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# >>> PERCORSI DI DEFAULT ALLINEATI A BE/RagCode/audit
DEFAULT_DB   = Path("BE") / "RagCode" / "audit" / "audit.db"
DEFAULT_JSON = Path("BE") / "RagCode" / "audit" / "audit.jsonl"

def _connect(db_path: Path) -> sqlite3.Connection:
    "connessione al db"
    if not db_path.exists():
        print(f"[ERR] DB non trovato: {db_path}", file=sys.stderr)
        sys.exit(2)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def _ts(s: Optional[str]) -> Optional[int]:
    "time stamp"
    """Parsa 'YYYY-MM-DD' o epoch int/float a epoch int (UTC)."""
    if not s:
        return None
    s = s.strip()
    if s.isdigit():
        return int(s)
    try:
        dt = datetime.fromisoformat(s)  # 'YYYY-MM-DD' o 'YYYY-MM-DDTHH:MM:SS'
    except ValueError:
        print(f"[ERR] Data non valida: {s}", file=sys.stderr)
        sys.exit(2)
    return int(dt.timestamp())

def _iso(ts_utc: int) -> str:
    return datetime.utcfromtimestamp(int(ts_utc)).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------

def cmd_events(args: argparse.Namespace):
    cx = _connect(args.db)
    q = """SELECT id, ts_utc, route, question_id, answer_status, model, latency_ms, question
           FROM audit_event WHERE 1=1"""
    conds, params = [], []
    if args.route:
        conds.append("route = ?"); params.append(args.route)
    if args.status:
        conds.append("answer_status = ?"); params.append(args.status)
    if args.model:
        conds.append("model = ?"); params.append(args.model)
    if args.question_id:
        conds.append("question_id = ?"); params.append(args.question_id)
    if args.time_from:
        conds.append("ts_utc >= ?"); params.append(_ts(args.time_from))
    if args.time_to:
        conds.append("ts_utc <= ?"); params.append(_ts(args.time_to))
    if conds:
        q += " AND " + " AND ".join(conds)
    q += " ORDER BY ts_utc DESC"
    if args.limit:
        q += f" LIMIT {int(args.limit)}"

    rows = cx.execute(q, params).fetchall()
    for r in rows:
        question_txt = (r["question"] or "").replace("\n", " ")
        qid = (r["question_id"] or "-")
        qid_short = qid[:8] if qid and qid != "-" else "-"
        print(
            f"{r['id']:>6}  {_iso(r['ts_utc'])}  {r['route']:<10}  qid={qid_short:<8}  "
            f"{(r['answer_status'] or '-'):<7}  {(r['model'] or '-')[:14]:<14}  "
            f"{str(r['latency_ms'] or '-'):<6}  {question_txt[:80]}"
        )

def cmd_show(args: argparse.Namespace):
    cx = _connect(args.db)
    ev = cx.execute("SELECT * FROM audit_event WHERE id=?", (args.id,)).fetchone()
    if not ev:
        print(f"[ERR] audit_event id {args.id} non trovato", file=sys.stderr); sys.exit(2)
    print("=== EVENT ===")
    for k in ev.keys():
        v = ev[k]
        if k == "ts_utc":
            v = f"{v} ({_iso(v)})"
        print(f"{k:>14}: {v}")
    print("\n=== META_JSON ===")
    try:
        meta = json.loads(ev["meta_json"] or "{}")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception:
        print(ev["meta_json"] or "-")
    print("\n=== RESOURCES ===")
    rs = cx.execute(
        """SELECT r.*
           FROM audit_resource r
           WHERE r.audit_id=?
           ORDER BY r.id ASC""",
        (args.id,),
    ).fetchall()
    if not rs:
        print("(nessuna risorsa)")
    else:
        for r in rs:
            print(
                f"- [{r['doc_id'] or '-'}] {r['doc_title'] or '-'} "
                f"(v={r['doc_version'] or '-'}, p={r['page']}, score={r['score']}, chunk={r['chunk_id'] or '-'})"
            )

def cmd_resources(args: argparse.Namespace):
    cx = _connect(args.db)
    if args.audit_id:
        q = """SELECT r.* FROM audit_resource r WHERE r.audit_id=? ORDER BY r.id"""
        rows = cx.execute(q, (args.audit_id,)).fetchall()
    else:
        q = """SELECT r.*, e.ts_utc, e.route, e.question_id
               FROM audit_resource r
               JOIN audit_event e ON e.id = r.audit_id
               WHERE 1=1"""
        conds, params = [], []
        if args.route:
            conds.append("e.route=?"); params.append(args.route)
        if args.question_id:
            conds.append("e.question_id=?"); params.append(args.question_id)
        if args.time_from:
            conds.append("e.ts_utc >= ?"); params.append(_ts(args.time_from))
        if args.time_to:
            conds.append("e.ts_utc <= ?"); params.append(_ts(args.time_to))
        if conds:
            q += " AND " + " AND ".join(conds)
        q += " ORDER BY e.ts_utc DESC, r.id ASC"
        if args.limit:
            q += f" LIMIT {int(args.limit)}"
        rows = cx.execute(q, params).fetchall()

    for r in rows:
        ts = r["ts_utc"] if "ts_utc" in r.keys() else None
        prefix = f"{_iso(ts)} " if ts else ""
        qid = (r["question_id"] if "question_id" in r.keys() else None) or "-"
        qid_short = qid[:8] if qid and qid != "-" else "-"
        print(
            f"{prefix}aid={r['audit_id']}  qid={qid_short:<8}  "
            f"[{r['doc_id'] or '-'}] {r['doc_title'] or '-'}  "
            f"(v={r['doc_version'] or '-'}, p={r['page']}, score={r['score']}, chunk={r['chunk_id'] or '-'})"
        )

def cmd_export(args: argparse.Namespace):
    cx = _connect(args.db)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # events
    q = "SELECT * FROM audit_event WHERE 1=1"
    conds, params = [], []
    if args.route: conds.append("route=?"); params.append(args.route)
    if args.status: conds.append("answer_status=?"); params.append(args.status)
    if args.model: conds.append("model=?"); params.append(args.model)
    if args.question_id: conds.append("question_id=?"); params.append(args.question_id)
    if args.time_from: conds.append("ts_utc>=?"); params.append(_ts(args.time_from))
    if args.time_to: conds.append("ts_utc<=?"); params.append(_ts(args.time_to))
    if conds: q += " AND " + " AND ".join(conds)
    q += " ORDER BY ts_utc DESC"
    ev_rows = cx.execute(q, params).fetchall()

    ev_csv = out_dir / "events.csv"
    with ev_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ev_rows[0].keys() if ev_rows else [
            "id","ts_utc","route","user_id","question_id","question",
            "answer_status","model","temperature","max_tokens","latency_ms","meta_json"
        ]
        w.writerow(headers)
        for r in ev_rows:
            w.writerow([r[h] for h in headers])

    # resources
    rs_rows = cx.execute(
        """SELECT r.* FROM audit_resource r
           WHERE r.audit_id IN (SELECT id FROM audit_event)"""
    ).fetchall()
    rs_csv = out_dir / "resources.csv"
    with rs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = rs_rows[0].keys() if rs_rows else [
            "id","audit_id","doc_id","doc_title","doc_version","page","score","chunk_id"
        ]
        w.writerow(headers)
        for r in rs_rows:
            w.writerow([r[h] for h in headers])

    print(f"[OK] Esportati {len(ev_rows)} eventi → {ev_csv}")
    print(f"[OK] Esportate {len(rs_rows)} risorse → {rs_csv}")

def cmd_stats(args: argparse.Namespace):
    cx = _connect(args.db)
    print("== STATISTICHE ==")
    # totali
    tot = cx.execute("SELECT COUNT(*) FROM audit_event").fetchone()[0]
    print(f"Eventi totali: {tot}")
    # per route
    print("\nPer route:")
    for r in cx.execute("SELECT route, COUNT(*) c FROM audit_event GROUP BY route ORDER BY c DESC"):
        print(f"- {r['route']}: {r['c']}")
    # per status
    print("\nPer status:")
    for r in cx.execute("SELECT COALESCE(answer_status,'NULL') s, COUNT(*) c FROM audit_event GROUP BY s ORDER BY c DESC"):
        print(f"- {r['s']}: {r['c']}")
    # latenza media per route
    print("\nLatenza media (ms) per route:")
    for r in cx.execute("SELECT route, ROUND(AVG(latency_ms),1) avg_ms, COUNT(*) c FROM audit_event WHERE latency_ms IS NOT NULL GROUP BY route ORDER BY avg_ms"):
        print(f"- {r['route']}: {r['avg_ms']} ms (n={r['c']})")
    # top documenti per frequenza
    print("\nTop 10 documenti (per frequenza in resources):")
    for r in cx.execute(
        """SELECT doc_id, doc_title, COUNT(*) c
           FROM audit_resource
           GROUP BY doc_id, doc_title
           ORDER BY c DESC
           LIMIT 10"""
    ):
        print(f"- [{r['doc_id'] or '-'}] {r['doc_title'] or '-'}  (n={r['c']})")

# --- JSONL mode (quick inspect) ---
def cmd_jsonl(args: argparse.Namespace):
    path = args.file or DEFAULT_JSON
    if not path.exists():
        print(f"[ERR] JSONL non trovato: {path}", file=sys.stderr); sys.exit(2)
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if args.route and obj.get("route") != args.route:
                continue
            if args.status and (obj.get("answer_status") or "") != args.status:
                continue
            if args.question_id and obj.get("question_id") != args.question_id:
                continue
            if args.contains and args.contains.lower() not in (obj.get("question") or "").lower():
                continue
            cnt += 1
            ts = obj.get("ts_utc")
            qtxt = (obj.get("question") or "").replace("\n", " ")
            qid = (obj.get("question_id") or "-")
            qid_short = qid[:8] if qid and qid != "-" else "-"
            print(
                f"{_iso(ts) if ts else '-'}  {obj.get('route')}  qid={qid_short:<8}  {obj.get('answer_status')}  "
                f"{(obj.get('model') or '-')[:14]:<14}  {qtxt[:80]}"
            )
            if args.limit and cnt >= args.limit:
                break
    if cnt == 0:
        print("(nessun match)")

# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Audit inspector (SQLite/JSONL)")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Percorso DB SQLite (default: {DEFAULT_DB})")

    sub = p.add_subparsers(dest="cmd", required=True)

    # events
    sp = sub.add_parser("events", help="Lista eventi (filtrabile)")
    sp.add_argument("--route", choices=["/search","/ask","/ask/stream","/fallback"], help="Filtro route")
    sp.add_argument("--status", choices=["OK","KO","PARTIAL"], help="Filtro status")
    sp.add_argument("--model", help="Filtro modello LLM")
    sp.add_argument("--question-id", help="Filtro question_id")  # 🆕
    sp.add_argument("--time-from", help="Da data (YYYY-MM-DD) o epoch")
    sp.add_argument("--time-to", help="A data (YYYY-MM-DD) o epoch")
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_events)

    # show
    sp = sub.add_parser("show", help="Mostra un evento + meta + resources")
    sp.add_argument("id", type=int, help="audit_event.id")
    sp.set_defaults(func=cmd_show)

    # resources
    sp = sub.add_parser("resources", help="Lista risorse (per audit_id o filtro route/tempo)")
    sp.add_argument("--audit-id", type=int)
    sp.add_argument("--route", choices=["/search","/ask","/ask/stream","/fallback"])
    sp.add_argument("--question-id", help="Filtro question_id")  # 🆕
    sp.add_argument("--time-from")
    sp.add_argument("--time-to")
    sp.add_argument("--limit", type=int, default=100)
    sp.set_defaults(func=cmd_resources)

    # export
    sp = sub.add_parser("export", help="Esporta CSV (events.csv, resources.csv)")
    sp.add_argument("--out", type=Path, required=True, help="Cartella output")
    sp.add_argument("--route", choices=["/search","/ask","/ask/stream","/fallback"])
    sp.add_argument("--status", choices=["OK","KO","PARTIAL"])
    sp.add_argument("--model")
    sp.add_argument("--question-id", help="Filtro question_id")  # 🆕
    sp.add_argument("--time-from")
    sp.add_argument("--time-to")
    sp.set_defaults(func=cmd_export)

    # stats
    sp = sub.add_parser("stats", help="Statistiche rapide")
    sp.set_defaults(func=cmd_stats)

    # jsonl
    sp = sub.add_parser("jsonl", help="Ispeziona audit.jsonl (append-only)")
    sp.add_argument("--file", type=Path, default=DEFAULT_JSON)
    sp.add_argument("--route", choices=["/search","/ask","/ask/stream","/fallback"])
    sp.add_argument("--status", choices=["OK","KO","PARTIAL"])
    sp.add_argument("--question-id", help="Filtro question_id")  # 🆕
    sp.add_argument("--contains", help="Filtro substring su question")
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_jsonl)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

# # ultimi 20 eventi
# python BE/RagCode/audit/audit_inspect.py events --limit 20

# # raggruppo per una domanda specifica
# python BE/RagCode/audit/audit_inspect.py events --question-id 9d1f2a4b

# # risorse per quella domanda
# python BE/RagCode/audit/audit_inspect.py resources --question-id 9d1f2a4b

# # export filtrato
# python BE/RagCode/audit/audit_inspect.py export --out BE/RagCode/audit/export --route /ask --question-id 9d1f2a4b

# # jsonl filter
# python BE/RagCode/audit/audit_inspect.py jsonl --question-id 9d1f2a4b --limit 30
