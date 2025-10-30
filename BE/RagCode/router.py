# router.py
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils import (
    embed_question,
    genera_motivazioni_llm,
    load_index_faq,
    load_metadata,
    search_by_text,
    build_prompt,
    call_llm,
    rerank_documents,
    chat_client,
    async_client_chat,
)

from audit.audit import start_event, add_resources, finish_event
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # /app

FAQ_INDEX = BASE_DIR / "AllVectorDB" / "FAQ" / "FAQ.index"
FAQ_META  = BASE_DIR / "AllVectorDB" / "FAQ" / "meta.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _map_resources_from_reranked(items: List[dict]) -> List[dict]:
    """
    Normalizza i campi per audit_resource:
    id/doc_id, title/doc_title, version/doc_version, page, score, chunk_id
    """
    out = []
    for s in items or []:
        out.append({
            "id": s.get("id") or s.get("doc_id"),
            "title": s.get("title") or s.get("doc_title"),
            "version": s.get("version") or s.get("doc_version"),
            "page": s.get("page"),
            "score": float(s.get("score")) if s.get("score") is not None else None,
            "chunk_id": s.get("chunk_id") or s.get("id"),
        })
    return out


def _extract_question_id(request: Request, payload_qid: Optional[str] = None) -> Optional[str]:
    """Prende il question_id dal payload o dall'header X-Question-Id (se presente)."""
    return payload_qid or request.headers.get("X-Question-Id") or None


# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------
logger = logging.getLogger("rag_router")
logger.propagate = True   # passa i log al root configurato in main.py
logger.setLevel(logging.INFO)



# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
rag_router = APIRouter()


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
class AskQuestion(BaseModel):
    question: str = Field(...)
    documents: List = Field(..., description="Sono la lista dei documenti selezionati")
    temperature: float = Field(0.2)
    max_tokens: int = Field(500)
    chat_model: str = Field("gpt-4o-mini", description="esplicitare il modello")
    question_id: Optional[str] = Field(None, description="ID logico della domanda (thread)")


class SearchDoc(BaseModel):
    question: str = Field(...)
    top_k: int = Field(5)
    chat_model: str = Field("gpt-4o-mini", description="esplicitare il modello")
    question_id: Optional[str] = Field(None, description="ID logico della domanda (thread)")


class SearchDocOutput(BaseModel):
    reranked_list: List[Any]
    explaination: str
    explain_obj: Dict[str, Any]


class FallbackOutput(BaseModel):
    original_question: str
    message: str
    suggerimenti: List[Dict[str, Any]]


# ---------------------------------------------------------------------
# /ask (sincrono)
# ---------------------------------------------------------------------
@rag_router.post("/ask")
def ask(payload: AskQuestion, request: Request):
    t0 = time.time()
    question    = payload.question
    temperature = payload.temperature
    max_tokens  = payload.max_tokens
    reranked    = payload.documents or []
    chat_model  = payload.chat_model
    qid         = _extract_question_id(request, payload.question_id)

    # --- AUDIT start ---
    audit_id = start_event(
        route="/ask",
        question=question,
        model=chat_model,
        temperature=temperature,
        max_tokens=max_tokens,
        meta={"input_docs": len(reranked)},
        question_id=qid,
    )

    try:
        if reranked:
            logger.info(f" Richiesta ASK — question='{question[:60]}...' ({len(reranked)} doc)")

            messages, links = build_prompt(question, reranked)
            answer = call_llm(messages, model=chat_model, temperature=temperature, max_tokens=max_tokens)

            logger.info(" Risposta generata con successo")

            # AUDIT: risorse consultate
            add_resources(audit_id, _map_resources_from_reranked(reranked))
            finish_event(
                audit_id,
                answer_status=("OK" if (answer and answer.strip().upper() != "KO") else "KO"),
                latency_ms=int((time.time()-t0)*1000),
                meta_update={"links": links, "answer_len": len(answer or "")}
            )

            return {
                "answer": answer,
                "sources": reranked,
                "links": links,
                "steps": [],
                "images": [],
                "warnings": [],
            }
        else:
            # nessun documento -> KO
            add_resources(audit_id, [])
            finish_event(
                audit_id,
                answer_status="KO",
                latency_ms=int((time.time()-t0)*1000),
                meta_update={"reason": "no_documents"}
            )
            return {
                "answer": "KO",
                "sources": [],
                "links": [],
                "steps": [],
                "images": [],
                "warnings": [],
            }
    except Exception as e:
        logger.exception(" Errore in /ask")
        finish_event(
            audit_id,
            answer_status="KO",
            latency_ms=int((time.time()-t0)*1000),
            meta_update={"error": str(e)}
        )
        raise


# ---------------------------------------------------------------------
# /ask/stream (asincrono)
# ---------------------------------------------------------------------
@rag_router.post("/ask/stream")
async def ask_stream(payload: AskQuestion, request: Request):
    """
    Streaming compatibile con OpenAI SDK 2.6.0.
    - Usa SOLO `content.delta` (no snapshot `chunk`) per evitare duplicazioni.
    - I token al FE sono inviati come JSON: {"token": "<testo>"}.
    """
    t0 = time.time()
    question    = payload.question
    temperature = payload.temperature
    max_tokens  = payload.max_tokens
    reranked    = payload.documents or []
    model       = payload.chat_model
    qid         = _extract_question_id(request, payload.question_id)

    # --- AUDIT start ---
    audit_id = start_event(
        route="/ask/stream",
        question=question,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        meta={"input_docs": len(reranked)},
        question_id=qid,
    )

    def sse_json(event: str, obj: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"

    if not reranked:
        # chiusura audit immediata come KO + SSE minimale
        finish_event(
            audit_id,
            answer_status="KO",
            latency_ms=int((time.time()-t0)*1000),
            meta_update={"reason": "no_documents"}
        )

        async def gen_ko():
            yield sse_json("token", {"token": "KO"})
            yield sse_json("done", {"ok": True})

        return StreamingResponse(gen_ko(), media_type="text/event-stream")

    messages, links = build_prompt(question, reranked)
    logger.info(" Avvio stream — '%s...' (docs=%d)", question[:70], len(reranked))

    async def generate():
        token_count = 0
        # meta iniziale
        yield sse_json("meta", {"links": links, "sources": reranked})

        try:
            async with async_client_chat.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                async for event in stream:
                    if event.type == "content.delta":
                        token = getattr(event, "delta", None)
                        if token:
                            token_count += len(token)
                            yield sse_json("token", {"token": token})
                    elif event.type == "response.completed":
                        logger.info(" Stream completato")
                        break
                    elif event.type == "response.error":
                        err_msg = getattr(event, "error", None)
                        logger.error(" Errore stream: %s", err_msg)
                        yield sse_json("error", {"error": str(err_msg)})
                        break
                    else:
                        logger.debug("Evento ignorato: %s", event.type)
        except Exception as e:
            logger.exception(" Eccezione durante lo stream")
            yield sse_json("error", {"error": str(e)})
        finally:
            # --- AUDIT: chiusura evento ---
            add_resources(audit_id, _map_resources_from_reranked(reranked))
            finish_event(
                audit_id,
                answer_status=("OK" if token_count > 0 else "KO"),
                latency_ms=int((time.time()-t0)*1000),
                meta_update={"links": links, "token_count": token_count}
            )

            yield sse_json("done", {"ok": True})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------
@rag_router.post("/search")
def search_doc(payload: SearchDoc, request: Request) -> SearchDocOutput:
    t0 = time.time()
    app = request.app
    index = app.state.vdb_index
    meta  = app.state.vdb_meta
    qid   = _extract_question_id(request, payload.question_id)

    logger.info(f" Ricerca: '{payload.question[:80]}...' (top_k={payload.top_k})")

    # --- AUDIT start ---
    audit_id = start_event(
        route="/search",
        question=payload.question,
        model="text-embedding-3-small",  # (retriever) aggiorna se cambi
        meta={"top_k": int(payload.top_k), "chat_model": payload.chat_model},
        question_id=qid,
    )

    try:
        results = search_by_text(index, meta, payload.question, top_k=payload.top_k)

        reranked, explanation_md, explain_obj = rerank_documents(
            payload.question,
            results,
            client=chat_client,
            max_return=min(payload.top_k, 4),
            temperature=0.0,
            model=payload.chat_model
        )

        logger.info(f" Restituiti {len(reranked)} documenti dopo rerank")
        logger.info(f"⏱ Ricerca {(time.time()-t0)*1000:.0f} ms")

        # --- AUDIT: risorse consultate ---
        add_resources(audit_id, _map_resources_from_reranked(reranked))
        finish_event(
            audit_id,
            answer_status="OK",
            latency_ms=int((time.time()-t0)*1000),
            meta_update={"explain_len": len(explanation_md or ""), "items": len(reranked)}
        )

        return SearchDocOutput(
            reranked_list=reranked, explaination=explanation_md, explain_obj=explain_obj
        )
    except Exception as e:
        logger.exception(" Errore in /search")
        finish_event(
            audit_id,
            answer_status="KO",
            latency_ms=int((time.time()-t0)*1000),
            meta_update={"error": str(e)}
        )
        raise


# ---------------------------------------------------------------------
# /fallback
# ---------------------------------------------------------------------
@rag_router.post("/fallback")
def fallback_question(request: Request, question: str, question_id: Optional[str] = None) -> FallbackOutput:
    t0 = time.time()
    qid = _extract_question_id(request, question_id)

    # --- AUDIT start ---
    audit_id = start_event(
        route="/fallback",
        question=question,
        model="text-embedding-3-large",   # modello embedding FAQ della query
        question_id=qid,
    )

    try:
        q_vec = embed_question(question)
        index_faq = load_index_faq(str(FAQ_INDEX))

        D, I = index_faq.search(q_vec, 3)
        scores = D[0].tolist()
        ids    = [int(x) for x in I[0].tolist() if x != -1]

        metadata_faq = load_metadata(str(FAQ_META))

        suggerimenti: List[Dict[str, Any]] = []
        for idx, score in zip(ids, scores):
            if idx == -1:
                continue
            m = metadata_faq.get(str(idx))
            if not m:
                continue
            suggerimenti.append({
                "id": idx,
                "question": m.get("question", ""),
                "answer": m.get("answer", ""),
                "score": round(float(score), 3),
            })

        motivazioni = genera_motivazioni_llm(question, suggerimenti)
        for sugg, mot in zip(suggerimenti, motivazioni):
            sugg["motivazione"] = mot

        if suggerimenti and suggerimenti[0]["score"] >= 0.5:
            message = "Non ho trovato una fonte precisa. Prova una di queste domande correlate:"
        else:
            message = ("Nessuna corrispondenza forte trovata. Ecco alcune domande vicine che "
                       "possono aiutare a circoscrivere l’argomento:")

        # --- AUDIT: logga i suggerimenti come risorse (namespace 'faq:')
        add_resources(audit_id, [
            {
                "id": f"faq:{s['id']}",
                "title": s.get("question"),
                "version": None,
                "page": None,
                "score": s.get("score"),
                "chunk_id": None,
            } for s in suggerimenti
        ])
        finish_event(
            audit_id,
            answer_status="OK",
            latency_ms=int((time.time()-t0)*1000),
            meta_update={"count": len(suggerimenti)}
        )

        return FallbackOutput(
            original_question=question,
            message=message,
            suggerimenti=suggerimenti
        )
    except Exception as e:
        logger.exception(" Errore in /fallback")
        finish_event(
            audit_id,
            answer_status="KO",
            latency_ms=int((time.time()-t0)*1000),
            meta_update={"error": str(e)}
        )
        raise


@rag_router.post("/warning")
def warning(request: Request, answer: str, question: str, question_id: str) -> Dict[str, Any]:
    """
    Analizza domanda e risposta e valuta se è necessario un warning tecnico o
    una nota di conformità, tracciando l'evento per audit.
    """
    # --- Extract Question ID ---
    qid = _extract_question_id(request, question_id)

    # --- AUDIT start ---
    audit_id = start_event(
        route="/warning",
        question=question,
        model="gpt-4o-mini",
        question_id=qid,
        meta={"phase": "post_answer_analysis"}
    )

    model_name = "gpt-4o-mini"

    SYSTEM_WARNING = """
    Sei un assistente di validazione tecnica e devi indicare se la risposta fornita
    richiede il supporto, la revisione o l'intervento di un esperto umano.

    Analizza in modo obiettivo e sintetico:

    1. Domanda: valuta se è di tipo operativo, tecnico, normativo o di sicurezza.
    2. Risposta: verifica se può comportare rischi, azioni su dispositivi o scelte che
       richiedono competenza professionale (es. manutenzione hardware, configurazioni
       avanzate, operazioni su stampanti, rete, firmware o documenti fiscali).
    3. Giudizio: indica se è opportuno aggiungere una nota di conformità o avvertenza.

    Rispondi SOLO in JSON:
    {
      "warning_needed": true | false,
      "message": "<messaggio sintetico>"
    }
    """

    prompt = f"""
    DOMANDA:
    {question}

    RISPOSTA:
    {answer}
    """

    try:
        response = chat_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_WARNING},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        text = response.choices[0].message.content.strip()

        try:
            result = json.loads(text)
        except Exception:
            result = {"warning_needed": False, "message": "Nessuna avvertenza necessaria"}

        # --- AUDIT SUCCESS ---
        finish_event(
            audit_id,
            answer_status="OK",
            latency_ms=None,
            meta_update={
                "warning_needed": result.get("warning_needed"),
                "message": result.get("message"),
                "question_id": qid,
            },
        )

        return {
            "question_id": qid,
            "warning_needed": result.get("warning_needed", False),
            "message": result.get("message", "Nessuna avvertenza necessaria"),
        }

    except Exception as e:
        logger.exception("Errore in /warning")

        finish_event(
            audit_id,
            answer_status="KO",
            latency_ms=None,
            meta_update={"error": str(e), "question_id": qid},
        )

        return {
            "question_id": qid,
            "warning_needed": False,
            "message": f"Errore durante la valutazione del warning: {e}",
        }

