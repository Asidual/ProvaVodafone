# utils.py
import os, json, logging, re, time
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("rag.utils")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
EMBED_MODEL_FAQ = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Per le immagini
STATIC_MOUNT = "/static".rstrip("/")
STATIC_MOUNT_DOC = "/staticdoc".rstrip("/")

chat_client = OpenAI(api_key=OPENAI_API_KEY)
async_client_chat = AsyncOpenAI(api_key=OPENAI_API_KEY)





_IMG_MD_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

#  Vector DB 
def _paths(db_dir: str | Path) -> Tuple[Path, Path]:
    db = Path(db_dir)
    return db / "index.faiss", db / "meta.json"

def load_vectordb(db_dir: str | Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    p_idx, p_meta = _paths(db_dir)
    if not p_idx.exists() or not p_meta.exists():
        raise FileNotFoundError(f"Vector DB non trovato in {db_dir}. Attesi: index.faiss e meta.json")
    index = faiss.read_index(str(p_idx))
    meta  = json.loads(p_meta.read_text(encoding="utf-8"))
    return index, meta

def norm_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12 #Usato per stabilizzare in caso ci fosse lo 0
    return x / norms

#  Embeddings 
def embed_text(text: str) -> np.ndarray:
    if not text:
        raise ValueError("Testo vuoto per embedding.")
    resp = chat_client.embeddings.create(model=EMBED_MODEL, input=text.strip())
    return np.array(resp.data[0].embedding, dtype="float32")

#  Search 
def search_by_text(index: faiss.Index, 
                   meta: List[Dict[str, Any]], 
                   query: str, 
                   top_k: int = 5) -> List[Dict[str, Any]]:
    
    q = norm_rows(embed_text(query).reshape(1, -1))
    D, I = index.search(q, top_k)
    D, I = D[0], I[0]
    results = []
    for score, idx in zip(D, I):
        if 0 <= idx < len(meta):
            item = dict(meta[idx])
            item["score"] = float(score)
            results.append(item)
    logger.info("search_by_text: '%s' → %d hits", query[:80].replace("\n", " "), len(results))
    return results

#  Utilities immagini 
def image_url_usage(url: str) -> str:
    if not url:
        logger.error("Non c'è nessun url valido")
        raise Exception
    
    if url.startswith("http://") or url.startswith("https://"):
        logger.info("Url valido: %s", url)
        return url
    
    # mount static
    return f"{STATIC_MOUNT}/{Path(url).name if '/' not in url else url.lstrip('/')}"

def extract_image_urls_from_chunk(c: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    # Lista di Figure interne (markdown)
    for item in c.get("Lista di Figure interne") or []:
        if isinstance(item, str):
            m = _IMG_MD_RE.search(item)
            if m:
                urls.append(m.group(2).strip())
            elif any(x in item for x in ("/", "\\")) or item.lower().endswith((".png",".jpg",".jpeg",".webp",".gif")):
                urls.append(item.strip())
        elif isinstance(item, dict):
            for k in ("image_path", "path", "link", "url"):
                v = item.get(k)
                if isinstance(v, str):
                    urls.append(v.strip()); break
    # Campi tipici dei record immagine
    if isinstance(c.get("images"), list):
        for p in c["images"]:
            if isinstance(p, str):
                urls.append(p.strip())
    if isinstance(c.get("image_path"), str):
        urls.append(c["image_path"].strip())

    # dedup preservando ordine
    seen=set(); out=[]
    for u in urls:
        if u and u not in seen:
            seen.add(u); out.append(u)
    return out

#  Prompt & LLM 
def build_prompt(question: str, contexts: List[Dict[str, Any]], *, max_chars: int = 2000):
    blocks, total_imgs, links = [], 0, []
    for i, c in enumerate(contexts, 1):
        ctype = c.get("Type") or "Text"
        page  = c.get("page") or c.get("Pagina")
        title = c.get("title") or c.get("Sezione ##") or c.get("chapter") or c.get("Capitolo #") or "(senza titolo)"
        raw   = (c.get("content") or "")
        text  = raw if len(raw) <= max_chars else (raw[:max_chars] + " …[troncato]")
        imgs  = extract_image_urls_from_chunk(c)
        total_imgs += len(imgs)
        block = [f"[{i}] Type: {ctype} | Titolo: {title} | Pagina: {page}", text]
        if imgs:
            block.append("Immagini:\n" + "\n".join(f"- {u}" for u in imgs))
        blocks.append("\n".join(block))
        
        if isinstance(page, list):
            for p in page:
                links.append(("PRINT!F",title,page,STATIC_MOUNT_DOC+f"/printf-manuale.pdf#page={p+1}"))
        elif isinstance(page,int):
            links.append(("PRINT!F",title,page,STATIC_MOUNT_DOC+f"/printf-manuale.pdf#page={page+1}"))
        else:
            continue

    system = (
        "Sei un assistente tecnico. Rispondi in ITALIANO, preciso e verificabile.\n"
        "Usa SOLO le fonti fornite; se non bastano, dillo esplicitamente.\n"
        "Cita con [indice] (es. [1]) e, se sono presenti immagini, riferisciti a ciò che mostrano senza inventare."
    )
    user = (
        f"Domanda: {question}\n\n"
        "Fonti:\n" + "\n\n".join(blocks) + "\n\n"
        "Istruzioni: fornisci una risposta sintetica, citando le fonti con [indice]. "
        "Se usi immagini, menzionale (es. 'vedi immagine in [2]')."
    )
    return [{"role":"system","content":system},{"role":"user","content":user}], links

def call_llm(messages: list, model:str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 500, ) -> str:
    resp = chat_client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return (resp.choices[0].message.content or "").strip()

# def call_llm_stream(messages, temperature, max_tokens):
#     """
#     Streamma i token in modo leggibile (aggiunge spazi mancanti).
#     """
#     stream = chat_client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stream=True,
#     )

#     last_token = ""
#     for chunk in stream:
#         delta = getattr(chunk.choices[0], "delta", None)
#         if not delta or not getattr(delta, "content", None):
#             continue

#         token = delta.content
        

#         # --- aggiusta la spaziatura ---
#         if (
#             last_token
#             and not last_token.endswith((" ", "\n", "-", "(", "[", "{", '"'))
#             and not token.startswith((" ", ".", ",", ";", ":", "!", "?", ")", "]", "}", "\n", '"'))
#         ):
#             token = " " + token
#             logger.info(f"Questo è il token: {token}")

#         yield token
#         last_token = token[-1] if token else last_token





#  Post-processing in payload strutturato 
_STEP_RE = re.compile(r"^\s*(\d+)[\).\s-]+\s*(.+)$")

def _parse_steps_from_answer(answer: str) -> List[Dict[str, Any]]:
    steps=[]
    for line in (answer or "").splitlines():
        m = _STEP_RE.match(line)
        if m:
            n = int(m.group(1))
            steps.append({"n": n, "text": m.group(2).strip(), "sources": []})
    # se non ci sono liste numerate, crea 1 passo unico
    if not steps and answer:
        steps=[{"n":1,"text":answer.strip(), "sources": []}]
    # ordina per n
    steps.sort(key=lambda x: x["n"])
    return steps

def _make_source_id(i:int) -> str:
    return f"S{i}"

def _first_nonempty(*vals):
    for v in vals:
        if v: return v
    return None

def build_structured_payload(
    question: str,
    contexts: List[Dict[str, Any]],
    llm_answer: str,
) -> Dict[str, Any]:
    """
    Converte (question, contexts, llm_answer) in payload RAG strutturato:
    - answer, steps, images, sources, warnings, fallback
    """
    # 1) SOURCES (identificatori S1, S2…)
    sources=[]
    all_image_urls=[]
    for i, c in enumerate(contexts, 1):
        sid = _make_source_id(i)
        page = c.get("page") or c.get("Pagina")
        title = _first_nonempty(c.get("title"), c.get("Sezione ##"), c.get("chapter"), c.get("Capitolo #"), "(senza titolo)")
        # estrai immagini e rendi URL assoluti
        imgs = [ image_url_usage(u) for u in extract_image_urls_from_chunk(c) ]
        all_image_urls.extend(imgs)
        # snippet quotabile
        quote = (c.get("content") or "")[:300]
        # link pagina (se hai un viewer PDF, sostituisci con URL viewer+anchor)
        page_link = None  # opzionale: costruisci qui un link tipo /pdf/Manuale?p=6
        sources.append({
            "id": sid,
            "title": title,
            "document": c.get("document") or "Manuale_PrintF.pdf",
            "page": page,
            "quote": quote,
            "link": page_link,
            "images": imgs,  # utile anche lato FE
            "Type": c.get("Type"),
            "score": c.get("score"),
        })

    # 2) IMAGES (lista piatta con id per FE)
    images=[]
    for i, url in enumerate(dict.fromkeys(all_image_urls), 1):  # dedup preservando ordine
        images.append({
            "id": f"S-IMG{i}",
            "url": url,
            "doc": "Manuale_PrintF.pdf",
            # se conosci bbox: "bbox": {"x":..,"y":..,"w":..,"h":..}
        })

    # 3) ANSWER e STEPS
    answer = (llm_answer or "").strip()
    steps  = _parse_steps_from_answer(answer)

    # 4) Collega fonti ai passi se il testo contiene [k]
    #    (rilevazione semplice: associa tutte le [n] trovate nella riga)
    bracket_ref = re.compile(r"\[(\d+)\]")
    for s in steps:
        refs = [int(x) for x in bracket_ref.findall(s["text"])]
        s["sources"] = [ _make_source_id(n) for n in refs if 1 <= n <= len(sources) ]

    # 5) Warnings (semplice euristica)
    warnings=[]
    kw = ("operazione fiscale", "supervisore", "assistenza", "sicurezza", "avvertenza", "omologata", "DGFE")
    if any(k.lower() in (c.get("content") or "").lower() for k in kw for c in contexts):
        warnings.append("Verifica eventuali requisiti fiscali e ruolo autorizzato prima di procedere.")
    # per domanda su spie, aggiungi qualcosa di sensato
    if "spia" in question.lower() or "spie" in question.lower():
        warnings.append("Se la spia 'Anomalia Stampante' permane accesa dopo i controlli base, spegnere e contattare assistenza autorizzata.")

    # 6) Fallback se answer è vuota: genera risposta descrittiva con immagini
    fallback = None
    if not answer:
        if images:
            # risposta “intelligente” che cita immagini
            lines = [
                "Non sono presenti informazioni testuali dirette nelle fonti, ma sono disponibili immagini di riferimento:",
            ]
            for i, im in enumerate(images, 1):
                lines.append(f"- [Immagine {i}] {im['url']}")
            lines.append("")
            lines.append("Vedi le miniature allegate e apri l’immagine per i dettagli.")
            answer = "\n".join(lines)
            # crea uno step unico
            steps = [{"n":1, "text":"Consulta le immagini di riferimento per identificare posizione e stato delle spie/indicatori. [1]", "sources":["S1"] if sources else []}]
        else:
            fallback = {
                "faqs": [
                    "Dove si trova il tasto Feed e cosa fa?",
                    "Come sostituire il rotolo di carta termica?",
                    "Cosa fare se la spia 'Anomalia Stampante' resta accesa?"
                ],
                "ask_clarification": "Vuoi che ti guidi nella sostituzione della carta o nel controllo del coperchio?"
            }
            answer = "Non ho trovato riferimenti sufficienti nelle fonti fornite. Vedi FAQ correlate qui sotto."

    return {
        "answer": answer,
        "steps": steps,
        "images": images,
        "sources": sources,
        "warnings": warnings,
        "fallback": fallback
    }
# ----- RERANK ------

# TODO aggiustare
RERANK_SYSTEM = (
    "Sei un assistente tecnico. Devi selezionare SOLO i documenti (chunk) strettamente pertinenti "
    "Analizza in dettaglio la domanda di riferimento e il contenuto. Valuta attentamente e in maniera critica selezionando solo quelle strettamente necessarie a rispondere alla domanda"
    "Rispondi solo con la documentazione disponibile senza fare riferimento a niente che non sia nella documentazione"
    "Se si presentano espressioni di cattiveria, odio o espressioni comuni non devi selezionare nessun documento"
    "Citazioni a Nomi o Cose che non sono inerenti con il tema delle stampanti che potrai vedere dai 'Documenti candidati' ritorna zero documenti selezionati "
    "Non inventare nulla. Se un chunk è un'IMMAGINE, considera solo ciò che mostra "
    "o la sua descrizione; se non utile, scartalo. Restituisci SOLO JSON valido."
)

def build_citation_line(i: int, doc: Dict[str, Any], max_chars: int = 1200) -> str:
    ctype = doc.get("Type")
    page = doc.get("page")
    title = doc.get("title") or doc.get("section") or doc.get("chapter")
    content = doc.get("content")
    score = doc.get("score")
    if max_chars and len(content)<max_chars:
        content = content[:max_chars] + " … [troncato]"
    imgs = []
    if isinstance(doc.get("images"), list): # Caso delle immagini
        imgs += [str(x) for x in doc["images"] if isinstance(x, str)]

    if isinstance(doc.get("Lista di Figure interne"), list): # Caso del testo
        imgs += [str(x) for x in doc["Lista di Figure interne"]]

    img_note = f"\nImmagini: {', '.join(imgs)}" if imgs else ""

    return f"[{i}] Type={ctype} | Pagina={page} | Titolo={title}\n{content}{img_note} | cosine_score={score}"

def explaination_selected_doc(question: str, 
                              selected: List[int], # Tipo [1, 2, 4]
                              items: List[Dict[str, Any]], 
                              cands: List[Dict[str, Any]]) -> str:
    
    # Scrivo come se fosse markdown per FE
    lines = [f"**Domanda:** {question}"]

    if not selected:
        lines.append("_Nessun documento strettamente pertinente individuato; forniti i migliori disponibili._")
        return "\n\n".join(lines)
    
    lines.append("**Motivazioni di selezione**")

    for it in items:
        idx = it.get("idx")
        why = it.get("why") or ""
        covers = it.get("covers") or []

        contrib_text = it.get("contribution_text") or ""
        contrib_img  = it.get("contribution_images") or ""
        pertinence = it.get("pertinence") or ""

        # dati di contesto
        if isinstance(idx, int) and 1 <= idx <= len(cands):
            doc = cands[idx-1]
            page = doc.get("page")
            title = doc.get("title") or doc.get("Sezione ##") or doc.get("chapter") or doc.get("Capitolo #")
            ctype = doc.get("Type")
            score = doc.get("score")

            lines.append(f"- **[{idx}] {title}** (Tipo: {ctype}; Pagina: {page}; Score Cosine: {score:3f})")

        else:

            lines.append(f"- **[{idx}]**")

        if why:
            lines.append(f"  - *Perché selezionato:* {why}")

        if covers:
            lines.append(f"  - *A copertura di:* {', '.join(covers)}")

        if contrib_text: 
            lines.append(f"  - *Contributo testo:* {contrib_text}")

        if contrib_img:  
            lines.append(f"  - *Contributo immagini:* {contrib_img}")

        if pertinence:
            lines.append(f"  - *Rivalutazione del chunk:* {pertinence}")

    return "\n".join(lines)

def rerank_documents(
    question: str,
    candidates: List[Dict[str, Any]],
    client: Optional[OpenAI] = None,
    model: str = None,
    max_return: int = 3,
    temperature: float = 0.0,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Usa l'LLM per scegliere i soli documenti pertinenti e spiega la scelta.
    Ritorna: (docs_scelti, explanation_markdown, explain_obj)
    """
    if not candidates:
        return [], "Nessun documento da valutare.", {"keep": [], "overview": "", "items": []}

    if not model:
        model = CHAT_MODEL
        # return [], "Non c'è un modello da usare per il reranking", {"keep": [], "overview": "", "items": []}

    # costruiamo lista citabile
    citation_lines = []
    for i, doc in enumerate(candidates, 1):
        citation_lines.append(build_citation_line(i, doc))

    user_prompt = (
        f"Domanda: {question}\n\n"
        "Documenti candidati:\n" + "\n\n".join(citation_lines) + "\n\n"
        "Istruzioni:\n"
        f"- Scegli SOLO i documenti utili e verificabili. Massimo numero di documenti: {max_return}.\n"
        "- Se non ci sono documenti pertinenti alla domanda non selezionare nessuno."
        "- Espressioni colloquiali non richiamano nessun documento quindi ritorna vuoto come 'Ciao', 'Come Va' ritorna {{}}"
        "- Espressioni di odio non devono selezionare nessun documento  ritorna {{}}"
        "- Se un'immagine non aggiunge nessun valore diretto, scartala.\n"
        "- Restituisci JSON con questo schema ESATTO:\n"
        "{\n"
        '  "keep": [1, 3, 5],\n'
        '  "overview": "spiegazione sintetica (1-3 frasi) del criterio di selezione e di come i documenti rispondono",\n'
        '  "items": [\n'
        '    {"idx": 1, "why": "motivo della scelta", "covers": ["sub-aspetto1","sub-aspetto2"],\n'
        '     "contribution_text": "quale parte testuale è utile (se rilevante)",\n'
        '     "contribution_images": "quali elementi visivi sono utili (se rilevante)"'
        '     "pertinence": "seleziona tra Eccellente, Ottimo, Buono, Disceto, Non Rilevante"}\n'
        '  ]\n'
        "}\n"
        "- Usa solo indici esistenti nei candidati (1..N)."
        "- La valutazione per pertinence indica la scala di rivalutazione fatta dal modello."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": RERANK_SYSTEM},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=480,
        )
        data = json.loads(resp.choices[0].message.content)
        logger.info(f"Questa è la risposta della selezione:\n{data}")
    except Exception:
        logger.exception("rerank_documents: errore LLM; uso top-K originali e nota di fallback.")
        sel = candidates[:max_return]
        expl = "Selezione di fallback: primi risultati di similarità (nessuna spiegazione LLM disponibile)."
        return sel, expl, {"keep": list(range(1, len(sel)+1)), "overview": expl, "items": []}
    if data.get("keep"): 
        keep = data.get("keep") or []
        items = data.get("items") or []
        overview = data.get("overview") or ""

        # sanificazione: interi in range, dedup, max_return
        selected_idx = []
        for k in keep:
            if isinstance(k, int) and 1 <= k <= len(candidates):
                if k not in selected_idx:
                    selected_idx.append(k)
            if len(selected_idx) >= max_return:
                break

        if not selected_idx:
            # ulteriore fallback in caso non venga selezionato nulla
            sel = candidates[:max_return]
            expl = "Nessun documento specifico indicato dal LLM; uso i primi risultati per coprire la risposta."
            return sel, expl, {"keep": [], "overview": overview, "items": items}

        selected_docs = [candidates[i-1] for i in selected_idx]
        explanation_md = explaination_selected_doc(question, selected_idx, items, candidates) # In markdown
        
        if overview:
            explanation_md = f"{explanation_md}\n\n**Sintesi:** {overview}"

        logger.info("rerank_documents → keep=%s", selected_idx)
        
        return selected_docs, explanation_md, {"keep": selected_idx, "overview": overview, "items": items}
    else:
        logger.info("Non sono stati trovati documenti inerenti alla domanda")
        return [], "", {}


# Fallback
def embed_question(text: str) -> np.ndarray:
    resp = chat_client.embeddings.create(model=EMBED_MODEL_FAQ, input=[text])
    vec = np.array([resp.data[0].embedding], dtype=np.float32)  # shape (1,D)
    vec_norm = vec/(np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12 )
    return vec_norm

def load_index_faq(index_path_faq:str) -> faiss.Index:
    return faiss.read_index(str(index_path_faq))

def load_metadata(meta_path_faq:str) -> dict:
    with open(meta_path_faq, "r", encoding="utf-8") as f:
        return json.load(f)
    
def genera_motivazioni_llm(original_q: str,
                              candidates: List[Dict[str, str]]) -> List[str]:
    """
    Genera una motivazione concisa (<= 1 frase, <= 160 caratteri) per ciascun candidato.
    Ritorna una lista di stringhe della stessa lunghezza di candidates.
    In caso di errore o mismatch, restituisce stringhe vuote per quelle posizioni.
    """
    if not candidates:
        return []

    # Prompt strutturato: chiediamo un JSON array di stringhe per robustezza.
    sys = (
        "Sei un assistente che aiuta l'utente a riformulare la domanda quando la ricerca non ha fonti precise. "
        "Per ogni domanda candidata, genera UNA sola frase molto breve (<=160 caratteri) che spieghi "
        "perché quella domanda aiuta ad approfondire il tema dell'utente. "
        "Rispondi SOLO con un JSON array di stringhe nell'ordine dato, senza testo extra."
    )
    # Costruiamo un contenuto minimale e serializzabile
    user_payload = {
        "domanda_utente": original_q,
        "candidate_domande_e_risposte": [
            {"question": c.get("question", ""), "answer": c.get("answer", "")[:500]}  #per avere il contesto
            for c in candidates
        ],
        "format": "JSON array di stringhe, una motivazione per ciascun candidato, stesso ordine."
    }

    try:
        resp = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        # Proviamo a parsare come JSON array di stringhe
        motivazioni = json.loads(content)
        if not isinstance(motivazioni, list):
            return [""] * len(candidates)
        # allinea la lunghezza
        if len(motivazioni) < len(candidates):
            motivazioni += [""] * (len(candidates) - len(motivazioni))
        elif len(motivazioni) > len(candidates):
            motivazioni = motivazioni[:len(candidates)]
        # troncature di sicurezza
        motivazioni = [str(m)[:160] if isinstance(m, str) else "" for m in motivazioni]
        return motivazioni
    except Exception:
        # In caso di fallimento LLM, ritorna stringhe vuote (frontend può gestire fallback)
        return [""] * len(candidates)




