# app.py
import json
import os
import io
from pathlib import Path
import time
import requests
from typing import List, Dict, Any, Optional
import streamlit as st
from PIL import Image
import uuid
from dotenv import load_dotenv

load_dotenv()


def new_question_id() -> str:
    return uuid.uuid4().hex


# CONFIG
st.set_page_config(page_title="RAG QA Vod", page_icon="💬", layout="wide")


API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000/api").rstrip("/") #in docker diventa backend

# NEW: base pubblica per il browser (se non messa, fallback a API_BASE_URL)
# Per immagini 
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "http://localhost:8000/api").rstrip("/") #

# root = senza /api
API_ROOT = API_BASE_URL[:-4] if API_BASE_URL.endswith("/api") else API_BASE_URL
PUBLIC_API_ROOT = PUBLIC_API_BASE_URL[:-4] if PUBLIC_API_BASE_URL.endswith("/api") else PUBLIC_API_BASE_URL


# Static pubblici (devono essere raggiungibili dal browser)
STATIC_BASE_URL = f"{PUBLIC_API_ROOT}/static".rstrip("/")
STATIC_BASE_URL_DOC = f"{PUBLIC_API_ROOT}/staticdoc".rstrip("/")

DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))

THUMB_W = int(os.getenv("THUMB_W", "220"))


# HELPERS
def normalize_image_public_url(url: str) -> str:
    """
    Normalizza QUALSIASI forma in /static/<file> (o URL assoluto).
    """
    if not url:
        return url
    u = url.strip()

    # URL assoluti -> ok
    if u.startswith("http://") or u.startswith("https://"):
        return u

    # /static/<file> -> ok
    if u.startswith("/static/"):
        return u

    # rimuovi eventuale prefisso figures2/ da path relativi
    u = u.lstrip("/")
    if u.startswith("figures2/"):
        u = u.split("/", 1)[1]

    return f"/static/{u}"


# API
def search_doc_explain(
    question: str, top_k: int, chat_model: str, question_id: Optional[str] = None
):
    url = f"{API_BASE_URL}/search"
    payload = {
        "question": question,
        "top_k": top_k,
        "chat_model": chat_model,
        "question_id": question_id,
    }
    headers = {"X-Question-Id": question_id} if question_id else {}
    r = requests.post(url, json=payload, headers=headers, timeout=90)
    r.raise_for_status()
    return r.json()


def ask_backend(
    question: str,
    documents: list,
    temperature: float,
    max_tokens: int,
    chat_model: str,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/ask"
    payload = {
        "question": question,
        "documents": documents,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_model": chat_model,
        "question_id": question_id,
    }
    headers = {"X-Question-Id": question_id} if question_id else {}
    r = requests.post(url, json=payload, headers=headers, timeout=90)
    r.raise_for_status()
    return r.json()


def ask_backend_stream(
    question: str,
    documents: list,
    temperature: float,
    max_tokens: int,
    chat_model: str,
    question_id: Optional[str] = None,
):
    """
    Riceve la risposta in streaming dal backend /ask/stream e yielda chunk.
    """
    url = f"{API_BASE_URL}/ask/stream"
    payload = {
        "question": question,
        "documents": documents,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_model": chat_model,
        "question_id": question_id,
    }
    headers = {"X-Question-Id": question_id} if question_id else {}
    with requests.post(
        url, json=payload, headers=headers, stream=True, timeout=300
    ) as r:  # >>> CHANGED
        r.raise_for_status()
        event = None
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line or raw_line.startswith(":"):
                continue
            line = raw_line.strip()
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
                continue
            if line.startswith("data:"):
                data_str = line.removeprefix("data:").strip()
                if event == "meta":
                    try:
                        yield {"type": "meta", "data": json.loads(data_str)}
                    except json.JSONDecodeError:
                        pass
                elif event == "token":
                    try:
                        payload = json.loads(data_str)
                        yield {"type": "token", "data": payload["token"]}
                    except json.JSONDecodeError:
                        yield {"type": "token", "data": data_str}
                elif event == "error":
                    yield {"type": "error", "data": data_str}
                elif event == "done":
                    yield {"type": "done"}
                    break


def ask_fallback(question: str, question_id: Optional[str] = None) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/fallback"
    params = {"question": question}
    headers = {"X-Question-Id": question_id} if question_id else {}
    r = requests.post(url, params=params, headers=headers, timeout=60)  # >>> CHANGED
    r.raise_for_status()
    return r.json()

def warning_gen(question: str, answer: str, question_id: Optional[str] = None) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/warning"
    params = {"question": question, "answer": answer}
    if question_id:
        params["question_id"] = question_id  # necessario perché FastAPI lo aspetta come arg
    headers = {"X-Question-Id": question_id} if question_id else {}

    r = requests.post(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()



def health_backend() -> Dict[str, Any]:
    out = {"api_base": API_BASE_URL, "root": API_ROOT}
    try:
        r = requests.get(API_ROOT, timeout=5)
        out["root_status"] = r.status_code
        try:
            out["root_json"] = r.json()
        except Exception:
            out["root_text"] = r.text[:400]
    except Exception as e:
        out["root_error"] = str(e)
    try:
        r = requests.options(f"{API_BASE_URL}/ask", timeout=5)
        out["ask_options"] = r.status_code
    except Exception as e:
        out["ask_error"] = str(e)
    return out


# Ottieni l'immagine in miniatura
def fetch_thumbnail(
    public_path_or_url: str, thumb_w: int = THUMB_W
) -> Optional[Image.Image]:
    try:
        full = public_path_or_url
        if full.startswith("/static/"):
            full = f"{PUBLIC_API_ROOT}{full}"
        r = requests.get(full, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w > thumb_w:
            scale = thumb_w / float(w)
            img = img.resize((thumb_w, int(h * scale)))
        return img
    except Exception:
        return None


def sources_map(sources: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {s.get("id"): s for s in sources if isinstance(s, dict) and s.get("id")}


def render_citation(sid: str, smap: Dict[str, Dict[str, Any]]) -> str:
    s = smap.get(sid)
    if not s:
        return f"[{sid}]"

    title = s.get("title")
    page = s.get("page")

    # Path al documento PDF statico
    doc_url = (
        f"{PUBLIC_API_ROOT}/printf-manuale.pdf"  # creo il path per il documento
    )

    # Se la pagina esiste, aggiungila come anchor per il PDF viewer
    link = f"{doc_url}#page={page}" if page else doc_url

    # Label descrittiva
    label = (
        f"{Path(doc_url).name} · p.{page} · {title}"
        if page
        else f"{Path(doc_url).name} · {title}"
    )

    return f"[{sid} — {label}]({link})"


def render_suggerimenti(fb: Dict[str, Any]):
    st.subheader("🧭 SUGGERIMENTI")
    msg = fb.get("message") or "Prova con una di queste domande correlate:"
    st.caption(msg)

    sugs = fb.get("suggerimenti") or []
    for s in sugs:
        q = s.get("question", "")
        mot = s.get("motivazione", "")
        # mostra solo domanda + motivazione breve
        st.markdown(f"- **{q}**  \n  _{mot}_")


# Modelli disponibili
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
if DEFAULT_CHAT_MODEL not in AVAILABLE_MODELS:
    DEFAULT_CHAT_MODEL = AVAILABLE_MODELS[0]


# SIDEBAR
with st.sidebar:
    st.header("⚙️ Impostazioni")
    api_base_in = st.text_input("API base URL (server-side)", API_BASE_URL)
    pub_api_base_in = st.text_input("Public API base URL (browser)", PUBLIC_API_BASE_URL)

    if api_base_in and api_base_in != API_BASE_URL:
        API_BASE_URL = api_base_in.rstrip("/")
        API_ROOT = API_BASE_URL[:-4] if API_BASE_URL.endswith("/api") else API_BASE_URL

    if pub_api_base_in and pub_api_base_in != PUBLIC_API_BASE_URL:
        PUBLIC_API_BASE_URL = pub_api_base_in.rstrip("/")
        PUBLIC_API_ROOT = PUBLIC_API_BASE_URL[:-4] if PUBLIC_API_BASE_URL.endswith("/api") else PUBLIC_API_BASE_URL
        STATIC_BASE_URL = f"{PUBLIC_API_ROOT}/static".rstrip("/")
        STATIC_BASE_URL_DOC = f"{PUBLIC_API_ROOT}/staticdoc".rstrip("/")

    st.code(
f"""API_BASE_URL = {API_BASE_URL}
PUBLIC_API_BASE_URL = {PUBLIC_API_BASE_URL}
API_ROOT = {API_ROOT}
PUBLIC_API_ROOT = {PUBLIC_API_ROOT}
STATIC_BASE_URL = {STATIC_BASE_URL}
STATIC_BASE_URL_DOC = {STATIC_BASE_URL_DOC}""",
                    language="bash",
                )
    
    top_k = st.number_input(
        "Top-K", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
    )
    max_tokens = st.number_input(
        "Max tokens", min_value=100, max_value=4000, value=DEFAULT_MAX_TOKENS, step=50
    )
    streaming_mode = st.checkbox("Abilita streaming risposta", value=False)
    model_choice = st.selectbox(
        "Modello LLM",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_CHAT_MODEL),
        key="chat_model",
    )

    if st.button("🔎 Health check backend"):
        st.json(health_backend())

# HEADER
st.title("💬 RAG QA Console")
st.caption(
    "Chiedi in linguaggio naturale: il sistema cerca nel Vector DB, usa immagini e cita le fonti."
)

# STATE
if "history" not in st.session_state:
    st.session_state["history"] = []
if "current_qid" not in st.session_state:
    st.session_state["current_qid"] = None

# FORM
st.divider()
with st.form("qa_form", clear_on_submit=False):
    question = st.text_area(
        "Domanda",
        placeholder='Es. Dove sono le spie "Fine Carta", "Anomalia Stampante" e "Power"?',
    )
    submitted = st.form_submit_button("Chiedi")

# ACTION
if submitted and question and question.strip():
    qid = new_question_id()
    st.session_state["current_qid"] = qid
    t0 = time.time()
    try:
        search_result = search_doc_explain(
            question.strip(), int(top_k), model_choice, question_id=qid
        )
        reranked, explanation_md, explain_obj = (
            search_result.get("reranked_list"),
            search_result.get("explaination"),
            search_result.get("explain_obj"),
        )
        if not reranked:
            fb = ask_fallback(question.strip(), qid)
            print(fb)
            render_suggerimenti(fb)
            # History minimal
            st.session_state["history"].append(
                {
                    "question": question.strip(),
                    "answer": None,
                    "sources": [],
                    "images": [],
                }
            )
            # HISTORY
            if st.session_state["history"]:
                st.divider()
                st.subheader("Cronologia")
                for i, item in enumerate(reversed(st.session_state["history"]), 1):
                    st.markdown(f"**Q{i}**: {item['question']}")
                    with st.expander("Apri risposta"):
                        if item.get("answer"):
                            st.write(item["answer"])
                        if item.get("steps"):
                            st.markdown("**Piano operativo:**")
                            for s in item["steps"]:
                                st.markdown(f"- {s.get('n')}. {s.get('text')}")
                        if item.get("images"):
                            st.markdown("**Immagini:**")
                            for u in item["images"]:
                                st.markdown(f"- {u}")

            st.stop()  # <-- mostra solo fallback + history

        if streaming_mode:
            # st.subheader("💬 Risposta (streaming)")
            placeholder = st.empty()
            answer_accum = ""
            last_update = time.time()

            # Mostra anche i link appena ricevuti
            links = []
            first = True
            for chunk in ask_backend_stream(
                question.strip(),
                reranked,
                float(temperature),
                int(max_tokens),
                model_choice,
                question_id=qid,
            ):
                tnow = time.time()
                ttft = (tnow - t0) * 1000

                if chunk["type"] == "token":
                    token = chunk["data"]
                    # print("Token:",token)
                    answer_accum += token
                    # print("Answer:",answer_accum)
                    # ✅ aggiorna ogni 50ms per non bloccare Streamlit
                    if (tnow - last_update) > 0.05:
                        placeholder.markdown(answer_accum)
                        last_update = tnow

                elif chunk["type"] == "meta":
                    meta_links = chunk["data"].get("links", [])
                    if meta_links:
                        links.extend(meta_links)
                        # st.caption(" · ".join([f"[🔗 {link}]({API_ROOT+link})" for link in meta_links]))

                elif chunk["type"] == "done":
                    break
            # controllo KO su testo completo (oltre alla meta)
            # if "KO" in answer_accum.upper(): # SI può migliorare
            #     fb = ask_fallback(question.strip())
            #     render_suggerimenti(fb)
            #     st.success(f"Risposta (fallback) in {ttft:.0f} ms")
            #     # History minimal
            #     st.session_state["history"].append({
            #         "question": question.strip(),
            #         "answer": None,
            #         "sources": [],
            #         "images": [],
            #     })
            #     # HISTORY
            #     if st.session_state["history"]:
            #         st.divider()
            #         st.subheader("Cronologia")
            #         for i, item in enumerate(reversed(st.session_state["history"]), 1):
            #             st.markdown(f"**Q{i}**: {item['question']}")
            #             with st.expander("Apri risposta"):
            #                 if item.get("answer"):
            #                     st.write(item["answer"])
            #                 if item.get("steps"):
            #                     st.markdown("**Piano operativo:**")
            #                     for s in item["steps"]:
            #                         st.markdown(f"- {s.get('n')}. {s.get('text')}")
            #                 if item.get("images"):
            #                     st.markdown("**Immagini:**")
            #                     for u in item["images"]:
            #                         st.markdown(f"- {u}")
            #     st.stop()  # <-- mostra solo fallback + history

            # se NO KO, come prima
            st.subheader("💬 Risposta (streaming)")
            placeholder.markdown(answer_accum)

            data = {
                "answer": answer_accum,
                "sources": reranked,
                "links": links,
                "steps": [],
                "images": [],
                "warnings": [],
            }
            answer = data.get("answer")
            st.success(f"TTFT {ttft:.0f} ms")

        else:
            # st.subheader("💬 Risposta")
            # STANDARD MODE
            data = ask_backend(
                question.strip(),
                reranked,
                float(temperature),
                int(max_tokens),
                model_choice,
                question_id=qid
            )
            answer = data.get("answer") or ""
            # Se KO (status o testo), attiva FALLBACK e mostra SOLO SUGGERIMENTI + History
            # if "KO" in answer.upper():
            #     fb = ask_fallback(question.strip())
            #     render_suggerimenti(fb)
            #     st.success(f"Risposta (fallback) in {(time.time() - t0)*1000:.0f} ms")
            #     st.session_state["history"].append({
            #         "question": question.strip(),
            #         "answer": None,
            #         "sources": [],
            #         "images": [],
            #     })
            #     # HISTORY
            #     if st.session_state["history"]:
            #         st.divider()
            #         st.subheader("Cronologia")
            #         for i, item in enumerate(reversed(st.session_state["history"]), 1):
            #             st.markdown(f"**Q{i}**: {item['question']}")
            #             with st.expander("Apri risposta"):
            #                 if item.get("answer"):
            #                     st.write(item["answer"])
            #                 if item.get("steps"):
            #                     st.markdown("**Piano operativo:**")
            #                     for s in item["steps"]:
            #                         st.markdown(f"- {s.get('n')}. {s.get('text')}")
            #                 if item.get("images"):
            #                     st.markdown("**Immagini:**")
            #                     for u in item["images"]:
            #                         st.markdown(f"- {u}")
            #     st.stop()
            # else:
            st.subheader("Risposta")
            st.write(answer)

        st.success(f"Risposta ricevuta in {(time.time() - t0)*1000:.0f} ms")

        with st.expander("Response JSON (per il check)"):
            st.json(data)

        #TODO INSERIRE WARINING
        
        warn = warning_gen(question.strip(), answer, qid)
        if warn.get("warning_needed"):
            st.subheader("Warning", divider="orange")
            st.warning(warn.get("message", "Attenzione: verifica con un esperto."))


        # ---- ANSWER ----
        links = data.get("links")

        if links:
            st.subheader("Links")
            for link in links:
                st.markdown(
                    f"[[{link[0]}  -  Pag.{link[2]}  -  '{link[1]}']]({PUBLIC_API_ROOT+link[3]})",
                    unsafe_allow_html=True,
                )
        # ---- SOURCES MAP ----
        srcs = data.get("sources", []) or []
        smap = sources_map(srcs)

        # ---- STEPS ----
        # steps = data.get("steps", []) or []
        # if steps:
        #     st.subheader("Piano operativo")
        #     for s in sorted(steps, key=lambda x: x.get("n", 0)):
        #         n = s.get("n"); txt = s.get("text", "")
        #         refs = s.get("sources", []) or []
        #         st.markdown(f"**{n}.** {txt}")
        #         if refs:
        #             st.caption(" · ".join(render_citation(r, smap) for r in refs))

        # ---- IMAGES ----
        imgs = data.get("images", []) or []
        for s in srcs:
            for u in s.get("images") or []:
                imgs.append(
                    {
                        "id": f"{s.get('id')}-IMG",
                        "url": u,
                        "doc": s.get("document"),
                        "page": s.get("page"),
                    }
                )

        seen = set()
        dedup_imgs = []
        for im in imgs:
            u = im.get("url")
            if not u:
                continue
            public = normalize_image_public_url(
                u
            )  # -> "/static/<file>" oppure "http..."
            if public in seen:
                continue
            seen.add(public)
            link_url = public if public.startswith("http") else f"{PUBLIC_API_ROOT }{public}"
            dedup_imgs.append({**im, "url": public, "link_url": link_url})

        if dedup_imgs:
            st.subheader("Immagini")
            cols = st.columns(
                min(6, len(dedup_imgs))
            )  # più colonne = miniature più piccole
            thumb_px = 120  # <--- dimensione miniatura in pixel
            for i, im in enumerate(dedup_imgs):
                with cols[i % len(cols)]:
                    # usa HTML per rendere cliccabile l’immagine
                    st.markdown(
                        f"""
                        <a href="{im['link_url']}" target="_blank" rel="noopener">
                            <img src="{im['link_url']}" width="{thumb_px}" style="border-radius:8px; display:block; margin:4px auto;" />
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )
        # ---- SPIEGAZIONE ----
        if explanation_md:
            st.subheader("Perché questa risposta?")
            st.markdown(explanation_md)
        # ---- WARNINGS ----
        warnings = data.get("warnings", []) or []
        if warnings:
            st.subheader("Avvertenze / conformità")
            for w in warnings:
                st.warning(w)

        # # ---- FALLBACK ----
        # fb = data.get("fallback")
        # if fb:
        #     st.subheader("Suggerimenti")
        #     faqs = fb.get("faqs") or []
        #     for q in faqs: st.markdown(f"- {q}")
        #     if fb.get("ask_clarification"): st.info(fb["ask_clarification"])

        # ---- HISTORY ----
        st.session_state["history"].append(
            {
                "question": question.strip(),
                "answer": answer,
                # "steps": steps,
                "sources": srcs,
                "images": [im["link_url"] for im in dedup_imgs],
                # "warnings": warnings
            }
        )

    except requests.HTTPError as e:
        st.error(f"Errore HTTP: {e} — {getattr(e.response, 'text', '')}")
    except requests.RequestException as e:
        st.error(f"Errore di rete: {e}")
    except Exception as e:
        st.exception(e)

# HISTORY
if st.session_state["history"]:
    st.divider()
    st.subheader("Cronologia")
    for i, item in enumerate(reversed(st.session_state["history"]), 1):
        st.markdown(f"**Q{i}**: {item['question']}")
        with st.expander("Apri risposta"):
            if item.get("answer"):
                st.write(item["answer"])
            if item.get("steps"):
                st.markdown("**Piano operativo:**")
                for s in item["steps"]:
                    st.markdown(f"- {s.get('n')}. {s.get('text')}")
            if item.get("images"):
                st.markdown("**Immagini:**")
                for u in item["images"]:
                    st.markdown(f"- {u}")
