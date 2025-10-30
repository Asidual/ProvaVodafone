from typing import List, Literal, Tuple
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import re
from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# 1
def pdf_to_markdown(
    file_path: str,
    endpoint: str,
    api_key: str,
    model_id: str = "prebuilt-layout",
    starting_page: int = 1,
    split_by_page: bool = False,

) -> str | List[Tuple[int, str]]:
    """
    Converte un PDF in Markdown usando Azure AI Document Intelligence. Più facile per il tracciamento dei capitoli
    """
    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(api_key))

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=AnalyzeDocumentRequest(bytes_source=f.read()),
            output_content_format="markdown",   
        )
    result = poller.result()

    # Markdown completo in un'unica stringa
    full_md: str = result.content or ""
    if starting_page > 1:
        pages = result.pages
        # uniamo solo i contenuti (spans) delle pagine desiderate
        parts = []
        for p in pages:
            if p.page_number >= starting_page and p.spans:
                start = p.spans[0].offset
                length = p.spans[0].length
                parts.append(full_md[start:start+length])
        return "".join(parts)

    
    return full_md, result



# NORMALIZZAZIONE DEI CAPITOLI
import re

def normalize_numeric_headings(
    md_text: str,
    min_level: int = 1,
    max_level: int = 6,
) -> str:
    """
    Uniforma i livelli dei titoli Markdown in base al prefisso numerico.
    Regola: livello = (numero di punti) + 1, clamp tra min_level e max_level.
    Esempi: "1" -> #, "2.1" -> ##, "2.2.1" -> ###

    - Normalizza SOLO le righe che sono già heading (#...).
    """

    # --- Heading già presenti (# ...):
    # Nota: in re.VERBOSE, '#' va escapato come '\#'
    pat_heading = re.compile(
        r"""^(?P<indent>[ \t]{0,3})        # indent opzionale nel titolo (0-3 spazi/tab)
             (?P<hashes>\#{1,6})           # uno o più '#' per identificare il livello
             [ \t]*                        # spazi
             (?P<num>\d+(?:\.\d+)*)        # numero gerarchico (1, 2.1, 3.4.5)
             \b                            
             [ \t]+                        # almeno uno spazio
             (?P<title>.*\S)?              # resto del titolo (se presente)
             [ \t]*$                       # spazi finali
        """,
        re.VERBOSE,
    )

    lines = (md_text or "").splitlines()
    out = []

    for line in lines:
        # 1) prova a normalizzare heading già marcati con #
        m = pat_heading.match(line) # Identifica un possibile pattern riga per riga
        if m:
            indent = m.group("indent")
            num = m.group("num")
            title = m.group("title").strip()

            depth = num.count(".") + 1
            depth = max(min_level, min(max_level, depth))
            new_hashes = "#" * depth

            if title:
                out.append(f"{indent}{new_hashes} {num} {title}")
            else:
                out.append(f"{indent}{new_hashes} {num}")
            continue


        # altrimenti lascia la riga intatta
        out.append(line)

    return "\n".join(out)

# markdown = normalize_numeric_headings(markdown, 1, 3)




H_RE = re.compile(r"^(?P<indent>\s*)(?P<hashes>#{1,4})\s+(?P<title>.+)$", re.MULTILINE) # Regex che intercetta gli header da # a ####
NUMERIC_START_RE = re.compile(r"^\s*\(?\d+(?:\.\d+)*[.)-]?\s") # Regex che riconosce un titolo "numerato" (1., 2), 3-, 1.2.3, ecc.)

def demote_unnumbered_headers_to_bold(md_text: str) -> str:
    """
    - Se la riga è un header (#..####) e il titolo NON inizia con un numero -> rimuove # e mette **titolo**.
    - Se il titolo è numerato -> non cambia nulla.
    """
    def _repl(m: re.Match) -> str:
        title = m.group("title")
        if NUMERIC_START_RE.match(title):
            # Header numerato: lascio com'è
            return m.group(0)
        # Header non numerato: rimuovo i # e applico grassetto
        indent = m.group("indent") or ""
        return f"{indent}**{title.strip()}**"

    return H_RE.sub(_repl, md_text)

# markdown = demote_unnumbered_headers_to_bold(markdown)


# SPLITTING
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter

# --- regex e utility ---
BREAK_COMMENT_RE = re.compile(r"<!--\s*break[\s\S]*?-->", re.IGNORECASE)
PAGE_NUM_RE = re.compile(r'<!--\s*PageNumber\s*=\s*"(\d+)"\s*-->', re.IGNORECASE)

def fig_links_by_page(
    figs: List[Tuple],
    base_dir_alias: Optional[str] = None,
    page_offset: int = 0,
) -> Dict[int, List[str]]:
    """[(path,page)] -> {page: ['[name](link)', ...]} con offset di pagina."""
    by_page: Dict[int, List[str]] = {}
    for path, page in figs:
        name = Path(path).stem
        link = f"{base_dir_alias}/{Path(path).name}" if base_dir_alias else path
        key_page = max(1, page + page_offset)  # clamp a 1
        by_page.setdefault(key_page, []).append(f"[{name}]({link})")
    return by_page


def pages_from_text(text: str) -> List[int]:
    """Estrae tutti i PageNumber presenti (ordinati, unici)."""
    nums = [int(x) for x in PAGE_NUM_RE.findall(text or "")]
    return sorted(set(nums))

def page_field(pages: Optional[List[int]]):
    """int se una pagina, lista se >1, None se vuota."""
    if not pages:
        return None
    return pages[0] if len(pages) == 1 else pages

def split_markdown_build_records_by_page_markers_no_recursive(
    md_text: str,
    figs: List[Tuple[str, int]],
    headers_to_split_on = (("#", "chapter"), ("##", "section"), ("###", "subsection")),
    strip_headers: bool = False,
    figures_link_base: Optional[str] = None,
    figures_page_offset: int = -1,  # <-- nuovo default: sposta le figure alla pagina precedente
) -> List[Dict[str, Any]]:
    """
    Pipeline deterministica basata SOLO su PageNumber nel markdown:
      1) Rimuove <!--Break ...--> (mantiene PageNumber).
      2) Split per header (MarkdownHeaderTextSplitter) => un record per blocco.
      3) Pagine = marker PageNumber presenti nel blocco; se assenti, eredita dal blocco successivo (right-to-left).
      4) Figure = link per ciascuna pagina del blocco.
    """
    # 0) pulizia solo "Break"
    md_clean = BREAK_COMMENT_RE.sub("", md_text or "")

    # 1) split per header (niente splitter ricorrente)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=list(headers_to_split_on),
        strip_headers=strip_headers
    )
    blocks = md_splitter.split_text(md_clean)  # lista di Document

    # 2) prima passata: raccogli testo, meta e pagine grezze
    info = []
    for d in blocks:
        meta = d.metadata or {}
        chapter = meta.get("chapter")
        section = meta.get("section")
        text = d.page_content
        pages = pages_from_text(text)  # può essere []
        info.append({
            "Testo": text,
            "PagineList": pages,   # usata per ereditarietà
            "Capitolo #": chapter,
            "Sezione ##": section,
        })

    # 3) se vuoto, eredita dal successivo assumendo che se è vuoto eredita da quello più avanti
    next_pages: Optional[List[int]] = None
    for i in range(len(info) - 1, -1, -1):
        if info[i]["PagineList"]:
            next_pages = info[i]["PagineList"]
        else:
            info[i]["PagineList"] = next_pages or []

    # 4) indice figure per pagina
    figlinks_by_page = fig_links_by_page(
        figs,
        base_dir_alias=figures_link_base,
        page_offset=figures_page_offset,
    )

    # 5) costruzione records finali
    records = []
    n = 0
    for item in info:
        pages= item["PagineList"]
        fig_refs = []
        seen = set()
        for p in pages:
            for ref in figlinks_by_page.get(p, []):
                if ref not in seen:
                    fig_refs.append(ref)
                    seen.add(ref)

        n += 1
        records.append({
            "Testo": item["Testo"],
            "Pagina": page_field(pages),
            "Capitolo #": item["Capitolo #"],
            "Sezione ##": item["Sezione ##"],
            "Lista di Figure interne": fig_refs,
            "Global_order": n
        })

    return records

# records = split_markdown_build_records_by_page_markers_no_recursive(
#     md_text=markdown,                # markdown DI con <!-- PageNumber="N" -->
#     figs=figure_list,                # [(path, page), ...]
#     headers_to_split_on=(("#","chapter"),("##","section"),("###","subsection")),
#     strip_headers=False,
#     figures_link_base="figures2",    # opzionale
# )


def correct_fig_assigment(records:List[Dict]) -> List[Dict]:
    # Controllo fatto a mano con assegnazione della figura alla reale sezione
    records[3]["Lista di Figure interne"].append(records[4]["Lista di Figure interne"][0] )
    records[4]["Lista di Figure interne"]  = records[4]["Lista di Figure interne"][1:]
    records[9]["Lista di Figure interne"].append(records[10]["Lista di Figure interne"][0] )
    records[10]["Lista di Figure interne"]  = []
    records[11]["Lista di Figure interne"]  = []
    records[12]["Lista di Figure interne"]  = []
    records[18]["Lista di Figure interne"]  = [records[18]["Lista di Figure interne"][0]]
    records[19]["Lista di Figure interne"]  = [records[19]["Lista di Figure interne"][1]]
    records[20]["Lista di Figure interne"]  = [records[20]["Lista di Figure interne"][0]]
    records[21]["Lista di Figure interne"]  = [records[21]["Lista di Figure interne"][1]]
    records[27]["Lista di Figure interne"]  = [records[27]["Lista di Figure interne"][0]]
    records[28]["Lista di Figure interne"]  = [records[28]["Lista di Figure interne"][1]]
    records[29]["Lista di Figure interne"]  = [records[29]["Lista di Figure interne"][0]]
    records[30]["Lista di Figure interne"]  = [records[30]["Lista di Figure interne"][1]]
    records[31]["Lista di Figure interne"]  = [records[31]["Lista di Figure interne"][0]]
    records[32]["Lista di Figure interne"]  = [records[32]["Lista di Figure interne"][1]]
    records[33]["Lista di Figure interne"]  = [records[33]["Lista di Figure interne"][0]]
    records[34]["Lista di Figure interne"]  = [records[34]["Lista di Figure interne"][1]]
    records[35]["Lista di Figure interne"]  = [records[35]["Lista di Figure interne"][0]]
    records[36]["Lista di Figure interne"]  = [records[36]["Lista di Figure interne"][1]]
    records[37]["Lista di Figure interne"]  = [records[37]["Lista di Figure interne"][2]]
    records[39]["Lista di Figure interne"]  = []
    records[40]["Lista di Figure interne"]  = []
    records[41]["Lista di Figure interne"]  = []
    records[45]["Lista di Figure interne"]  = []
    records[47]["Lista di Figure interne"]  = []
    records[50]["Lista di Figure interne"]  = records[50]["Lista di Figure interne"][:2]
    records[51]["Lista di Figure interne"]  = records[51]["Lista di Figure interne"][2:]
    records[52]["Lista di Figure interne"]  = []
    records[54]["Lista di Figure interne"]  = []
    records[55]["Lista di Figure interne"]  = []
    records[57]["Lista di Figure interne"]  = []
    records[59]["Lista di Figure interne"]  = []
    records[60]["Lista di Figure interne"]  = []
    records[64]["Lista di Figure interne"]  = []
    records[65]["Lista di Figure interne"].append(records[66]["Lista di Figure interne"])
    records[66]["Lista di Figure interne"] = []
    records[67]["Lista di Figure interne"] = []
    records[68]["Lista di Figure interne"] = []
    records[75]["Lista di Figure interne"] = records[75]["Lista di Figure interne"][:1]
    records[76]["Lista di Figure interne"] = []
    records[77]["Lista di Figure interne"] = records[77]["Lista di Figure interne"][1:]
    records[78]["Lista di Figure interne"] = records[78]["Lista di Figure interne"][:1]
    records[79]["Lista di Figure interne"] = []
    records[80]["Lista di Figure interne"] = records[80]["Lista di Figure interne"][1:]
    records[81]["Lista di Figure interne"] = records[81]["Lista di Figure interne"][:1]
    records[82]["Lista di Figure interne"] = []
    records[83]["Lista di Figure interne"] = records[83]["Lista di Figure interne"][1:]
    records[84]["Lista di Figure interne"] = []
    records[86]["Lista di Figure interne"] = []
    records[87]["Lista di Figure interne"] = []
    return records



# EMBEDDING 

# inizializza client (assicurati di avere OPENAI_API_KEY nel tuo ambiente)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
#text-embedding-3-large

def embed_text_openai(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """
    Genera un vettore di embedding per una stringa di testo usando l'API OpenAI.
    """
    # Rimuove spazi bianchi superflui
    text = text.strip().replace("\n", " ")

    response = client.embeddings.create(
        input=text,
        model=model
    )

    # Estrae il vettore dal primo risultato
    embedding_vector = response.data[0].embedding
    return embedding_vector

def add_embedding(list_records: list)-> list:
    new_list = []
    for ele in list_records:
        vec = embed_text_openai(ele["Testo"])
        ele["vectorTesto"] = vec
        new_list.append(ele)
    return new_list

# ----------------------
# PROCESSAZIONE DELLE IMMAGINI 

from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import re
from typing import Dict, Any, List, Tuple, Optional, Iterable

def get_value_from_dict(d: Dict[str, Any], *keys, default=None):
    """Ritorna il primo d[k] esistente tra le chiavi fornite (gestisce camel/snake)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def extract_figures_png_robust_from_dict(
    pdf_path: str,
    di_dict: dict,
    out_dir: str = "figures2"
) -> List[Tuple[str, int]]:
    """
    Estrae le 'figure' da un dict ottenuto con AnalyzeResult.as_dict()
    (prebuilt-layout). Salva PNG in out_dir e ritorna [(percorso_png, pageNumber)].
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- Letture sicure dal dict (camelCase/snake_case) -----------------------
    pages = get_value_from_dict(di_dict, "pages", default=[]) # Estrae la lista di pages
    figures = get_value_from_dict(di_dict, "figures", default=[]) # Estrae la lista di figure

    # Mappa page_number -> info pagina
    def ottieni_page_number_from_sing_pages(p):  # int
        return int(get_value_from_dict(p, "pageNumber", default=0) or 0)

    di_pages = { ottieni_page_number_from_sing_pages(p): p for p in pages if ottieni_page_number_from_sing_pages(p) > 0 }

    # --- Apri il PDF una volta sola ------------------------------------------
    pdf = fitz.open(pdf_path)
    saved = []
    idx = 0

    for fig in figures:
        bregions = get_value_from_dict(fig, "boundingRegions", default=[]) 
        #Prende per ogni figura il bounding region che è una lista
        for br in bregions:
            pnum = int(get_value_from_dict(br, "pageNumber", default=0) or 0) # Da questo estraiamo il valore della pagina
            if pnum <= 0:
                continue

            di_page = di_pages.get(pnum) #Prendiamo solo dove è presente la pagina di riferimento
            polygon = get_value_from_dict(br, "polygon", default=None) # Prendi le info del poligono

            # servono almeno 4 combinazioni di (x,y)
            if (not di_page) or (not polygon) or (not isinstance(polygon, list)) or (len(polygon)//2 < 4):
                continue

            unit = (get_value_from_dict(di_page, "unit", default="pixel") or "pixel").lower()
            # width/height della pagina così come riportati da DI (in base a unit)
            di_w = float(get_value_from_dict(di_page, "width", default=0) or 0)
            di_h = float(get_value_from_dict(di_page, "height", default=0) or 0)

            page = pdf[pnum - 1]

            if unit == "pixel":
                # Renderizza la pagina alla risoluzione DI (pixel) per avere un 1:1
                if di_w <= 0 or di_h <= 0:
                    continue
                zoom_x = di_w / page.rect.width   # page.rect.* sono in punti (72 dpi)
                zoom_y = di_h / page.rect.height
                mat = fitz.Matrix(zoom_x, zoom_y)
                full_pix = page.get_pixmap(matrix=mat, alpha=False)
                # Converte il pixmap in PIL.Image per un crop “pixel perfect”
                img = Image.open(io.BytesIO(full_pix.tobytes("png")))

                xs = list(map(int, map(round, polygon[0::2])))
                ys = list(map(int, map(round, polygon[1::2])))
                x0, y0 = max(min(xs), 0), max(min(ys), 0)
                x1, y1 = min(max(xs), img.width), min(max(ys), img.height)
                if x1 <= x0 or y1 <= y0:
                    continue

                crop = img.crop((x0, y0, x1, y1))
                if crop.width * crop.height < 1:
                    continue

                idx += 1
                out_path = Path(out_dir) / f"page{pnum}_figure{idx}.png"
                crop.save(out_path)
                saved.append((out_path.as_posix(), pnum))

            if unit == "inch":
                # Dovrebbero essere tutte inches
                # Coordinate in pollici -> punti PDF (72 pt/in)
                pts = list(zip(polygon[0::2], polygon[1::2]))  # Coppie x, y
                xs = [float(x) * 72.0 for x, _ in pts]
                ys = [float(y) * 72.0 for _, y in pts]
                rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                if rect.width <= 0 or rect.height <= 0:
                    continue

                pix = page.get_pixmap(clip=rect, alpha=False)
                if pix.width * pix.height < 1: # per estremamente piccole no
                    continue

                idx += 1
                out_path = Path(out_dir) / f"page{pnum-1}_figure{idx}.png"
                pix.save(out_path.as_posix())
                saved.append((out_path.as_posix(), pnum))

            else:
                # Fallback: tratta come 'inch' se unità sconosciuta
                pts = list(zip(polygon[0::2], polygon[1::2]))
                xs = [float(x) * 72.0 for x, _ in pts]
                ys = [float(y) * 72.0 for _, y in pts]
                rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                if rect.width <= 0 or rect.height <= 0:
                    continue
                pix = page.get_pixmap(clip=rect, alpha=False)

                idx += 1
                out_path = Path(out_dir) / f"page{pnum}_figure{idx}.png"
                pix.save(out_path.as_posix())
                saved.append((out_path.as_posix(), pnum))

    pdf.close()
    return saved





FIG_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
def genera_lista_fig(fig_links: Any) -> Iterable[str]:
    """
    Normalizza 'Lista di Figure interne' in una sequenza di stringhe tipo:
      - "[name](path)"
    Accetta:
      - liste

    """
    # Se è iterabile (lista/tuple), scorri:
    if isinstance(fig_links, (list, tuple)):
        for item in fig_links:
            
            if isinstance(item, str):
                yield item
                continue
        return

    return

def extract_fig_names(fig_links: Any) -> List[Tuple[str, str]]:
    """
    Restituisce lista (name, link).
    Supporta:
      - "[name](link)"
      - plain path/URL (name = stem)
    """
    out= []
    for s in genera_lista_fig(fig_links):
        s = s.strip()
        # 1) caso markdown [name](link)
        m = FIG_LINK_RE.search(s)
        if m:
            name, link = m.group(1).strip(), m.group(2).strip()
            out.append((name, link))
            continue
        # 2) caso "solo path/url": prova a ricostruire
        if "://" in s or "/" in s or s.lower().endswith((".png")):
            link = s
            name = Path(s).stem
            out.append((name, link))
            continue
        # 3) caso "solo name": saltalo (senza path non è utile), oppure crea link neutro
        # out.append((s, s))  # sblocca se vuoi tenerlo comunque
    return out

def page_a_list(pagina_field) -> List[int]:
    # Lo trasforma in una lista dato che le pagine possono essere singole o più
    if pagina_field is None:
        return []
    if isinstance(pagina_field, int):
        return [pagina_field]
    seen = set()
    out = []
    for x in pagina_field:
        if isinstance(x, int) and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def build_figure_index_from_records(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Crea indice figura -> payload sezione/capitolo.
    Se una figura compare in più record (raro), tiene il PRIMO (ordine globale).
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for r in records:
        chapter = r.get("Capitolo #")
        section = r.get("Sezione ##")
        text = r.get("Testo") or ""
        pages = page_a_list(r.get("Pagina"))
        page = pages[0] if pages else None

        fig_links = r.get("Lista di Figure interne") # Sono delle liste
        for name, link in extract_fig_names(fig_links):
            if name in idx:
                continue
            idx[name] = {
                "chapter": chapter,
                "section": section,
                "text": text,
                "page": page,
                "image_link": link,
                "image_path": link,   # se locale, path = link
                "name": name,
            }
    return idx

def build_fig_items_from_records(
    records: List[Dict[str, Any]],
    figures_link_base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fig_idx = build_figure_index_from_records(records)
    out: List[Dict[str, Any]] = []

    for name, payload in fig_idx.items():
        img_path = payload["image_path"]
        # se vuoi forzare una base, sovrascrivi il path preservando il filename
        if figures_link_base:
            img_path = f"{figures_link_base}/{Path(img_path).name}"
        out.append({
            "name": name,
            "page": payload["page"],
            "chapter": payload["chapter"],
            "section": payload["section"],
            "text": payload["text"],
            "image_path": img_path,
            "image_link": f"[{name}]({img_path})",
        })

    out.sort(key=lambda x: (x["page"] if x["page"] is not None else 10**9, x["name"]))
    return out

# Hai già i records (uno per blocco) e la lista figure (path, page)
# fig_items = build_fig_items_from_records(
#     records_emd_official,
#     # figures=figure_list,
#     figures_link_base="figures2",
#     text_char_limit=None  # opzionale
# )#[2:]
# fig_items[i]["text"] = testo della sezione/capitolo corrispondente alla pagina della figura



import os, re, base64
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY non impostata"))
MODEL = os.getenv("OPENAI_VLM_MODEL", "gpt-4o")
CLIENT = OpenAI(api_key=OPENAI_API_KEY)


DEFAULT_BASE_DIRS = [
    Path.cwd(),
    # Path.cwd() / "figures2",
    # Path(r"C:\Users\laudi\OneDrive\Documenti\REPO\ProvaVodafone\BE\Notebook\figures2")

]

def _resolve_image(p: str, bases: Optional[Iterable[Path]] = None) -> Path:
    pth = Path(p)
    if pth.is_absolute() and pth.exists():
        return pth
    bases = list(bases) if bases else DEFAULT_BASE_DIRS #Usiamo fino al puntamento nell'ingestion
    tried = []
    for b in bases:
        cand = (b / pth).resolve()
        tried.append(cand)
        if cand.exists():
            return cand
        
    # fname = pth.name
    # for b in bases:
    #     hit = next(b.glob(f"**/{fname}"), None)
    #     if hit and hit.exists():
    #         return hit.resolve()
    raise FileNotFoundError(f"Immagine non trovata: {p}\nProvati:\n - " + "\n - ".join(map(str, tried)))

def _img_to_data_url(p: str) -> str:
    rp = _resolve_image(p)
    ext = rp.suffix.lower().lstrip(".") or "png" #Anche se sono tutti png
    mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
    return f"data:{mime};base64," + base64.b64encode(rp.read_bytes()).decode("utf-8")

# ===== Prompt “image-first” =====
SYSTEM_PROMPT = (
    "Sei un tecnico di manualistica specializzato in descrizioni VISIVE. "
    "PRIORITÀ: descrivi con precisione e ricchezza di dettagli SOLO ciò che è visibile nell'immagine. "
    "Il contesto (capitolo/sezione/testo) è secondario e serve solo per denominare elementi plausibili, senza inventare. "
    "Se qualcosa non è leggibile, dichiaralo. Evita ipotesi. "
    "Descrivi tutti i testi che sono presenti all'interno"
    "Rispondi con una sola descrizione discorsiva in italiano, senza preamboli o markup."
    "Separa se le immagini che rappresentano degli oggetti tangibili se sono più di uno, descrivendole in forma separata."
    ""
)

def user_prompt(*, chapter: Optional[str], section: Optional[str], page_text: Optional[str], language: str) -> str:
    ctx = []
    if chapter: 
        ctx.append(f"Capitolo: {chapter}")
    if section: 
        ctx.append(f"Sezione: {section}")
    if page_text: 
        ctx.append(f"Estratto: {(page_text)}")
    ctx_str = "\n".join(ctx) if ctx else "Nessun contesto aggiuntivo."
    return (
        f"[CONTESTO]\n{ctx_str}\n\n[ISTRUZIONI]\n"
        "1) Metti l'immagine al centro."
        "2) Descrivi componenti, disposizione, testi leggibili, icone/simboli, spie, porte/connettori, materiali/texture. "
        "3) Con più immagini, integra in un'unica descrizione evidenziando differenze." 
        "4) Se non leggibile/chiaro, dillo. "
        f"\n\n[STILE]\n- Lingua: {language}\n- Tono: tecnico, chiaro, conciso ma completo.\n\n"
        "Restituisci SOLO la descrizione finale (testo continuo)."
    )

# ===== Funzione principale: descrive 1+ immagini =====
def describe_figure(
    image_paths: List[str],
    *,
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    page_text: Optional[str] = None,
    language: str = "italiano",
    model: str = MODEL,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    user_content= [{"type": "text", "text": user_prompt(chapter=chapter, section=section, page_text=page_text, language=language)}]
    found = False
    for p in image_paths:
        try:
            user_content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(p)}})
            found = True
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
    if not found:
        raise FileNotFoundError("Nessuna immagine valida trovata nei path forniti.")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    resp = CLIENT.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
    return (resp.choices[0].message.content or "").strip()



GroupStrategy = Literal["per_page", "per_image"]

def describe_from_items(
    items: List[Dict[str, Any]],
    *,
    language: str = "italiano",
    temperature: float = 0.2,
    max_tokens: int = 700,
    group_strategy: GroupStrategy = "per_image", 
) -> List[Dict[str, Any]]:
    out = []

    if group_strategy == "per_image":
        # 1 descrizione per ogni item/immagine
        for it in items:
            img = it.get("image_path")
            if not img:
                continue
            desc = describe_figure(
                [img],
                chapter=it.get("chapter"),
                section=it.get("section"),
                page_text=it.get("text"),
                language=language,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out.append({
                "name": it.get("name"),
                "page": it.get("page"),
                "chapter": it.get("chapter"),
                "section": it.get("section"),
                "images": [img],
                "description": desc,
            })
        return out

    # altrimenti raggruppo per pagina (comportamento precedente)
    # by_page: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"images": [], "chapter": None, "section": None, "text": ""})
    # for it in items:
    #     page = it.get("page")
    #     if page is None:
    #         continue
    #     if it.get("image_path"):
    #         by_page[page]["images"].append(it["image_path"])
    #     if it.get("text"):
    #         by_page[page]["text"] += (("\n\n" if by_page[page]["text"] else "") + it["text"])
    #     by_page[page]["chapter"] = by_page[page]["chapter"] or it.get("chapter")
    #     by_page[page]["section"] = by_page[page]["section"] or it.get("section")

    # for page in sorted(by_page):
    #     b = by_page[page]
    #     desc = describe_figure(
    #         b["images"],
    #         chapter=b["chapter"],
    #         section=b["section"],
    #         page_text=b["text"],
    #         language=language,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #     )
    #     out.append({
    #         "page": page,
    #         "chapter": b["chapter"],
    #         "section": b["section"],
    #         "images": b["images"],
    #         "description": desc
    #     })
    return out

def add_embedding_image(list_records: list)-> list:
    new_list = []
    for ele in list_records:
        vec = embed_text_openai(ele["description"])
        ele["vectorTesto"] = vec
        new_list.append(ele)
    return new_list
# results_image_embedd = add_embedding_image(results)



### -- MERGING E FAISS --
def merge_text_and_image_records(text_records: list, image_records: list) -> list:
    """
    Unisce le due liste (testo e immagini) in una lista unica uniformata per l'indicizzazione.
    Aggiunge il campo 'Type' = 'Text' o 'Image' e conserva tutti gli altri campi.
    """
    unified = []

    # --- Uniforma i record di testo ---
    for t in text_records:
        unified.append({
            "Type": "Text",
            "page": t.get("Pagina"),
            "chapter": t.get("Capitolo #"),
            "section": t.get("Sezione ##"),
            "title": t.get("Sezione ##") or t.get("Capitolo #"),
            "content": t.get("Testo"),
            "vector": t.get("vectorTesto"),
            # conserva tutto il resto
            **{k: v for k, v in t.items() if k not in ("Testo", "vectorTesto")},
        })

    # --- Uniforma i record di immagine ---
    for img in image_records:
        unified.append({
            "Type": "Image",
            "page": img.get("page"),
            "chapter": img.get("chapter"),
            "section": img.get("section"),
            "title": img.get("name"),
            "content": img.get("description"),
            "vector": img.get("vectorTesto"),
            # conserva tutto il resto
            **{k: v for k, v in img.items() if k not in ("description", "vectorTesto", "vector")},
        })

    # --- Ordina per pagina se presente ---
    # unified.sort(key=lambda x: (x.get("page") if x.get("page") is not None else 1e9, x["Type"]))
    return unified

import hashlib
import json

def add_chunk_ids(chunks: list) -> list:
    """
    Aggiunge un ID univoco e stabile a ogni chunk della lista.
    L'ID è derivato da campi chiave e da un hash SHA1 del contenuto.
    Ritorna una nuova lista con la chiave 'id' aggiunta.
    """
    out = []

    for ch in chunks:
        # costruiamo l'identificativo basato su contesto e contenuto
        base_fields = [
            str(ch.get("Type", "")),
            str(ch.get("page", "")),
            str(ch.get("chapter", "")),
            str(ch.get("section", "")),
            str(ch.get("title", "")),
            str(ch.get("Global_order", "")),
        ]
        content_str = json.dumps(ch.get("content", ""), ensure_ascii=False)
        content_hash = hashlib.sha1(content_str.encode("utf-8")).hexdigest()[:10]
        raw_id = "|".join(base_fields) + "|" + content_hash
        chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]  # ID breve

        # aggiungi id al chunk
        ch_with_id = {**ch, "id": chunk_id}
        out.append(ch_with_id)

    return out

# merged_id = add_chunk_ids(merged)


import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

# ====== Utility ======
def _paths(db_dir: str | Path) -> Tuple[Path, Path]:
    db = Path(db_dir); db.mkdir(parents=True, exist_ok=True)
    return db / "index.faiss", db / "meta.json"

def _norm_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms.astype("float32")

def _as_vec(v) -> np.ndarray:
    return np.asarray(v, dtype="float32")

# ====== Creazione DB ======
def build_vectordb(db_dir: str | Path, chunks: List[Dict[str, Any]]) -> None:
    """
    Crea un Vector DB locale da una lista di chunk con campo 'vector'.
    Salva index.faiss (FAISS) e meta.json (metadati completi).
    """
    vectors, meta = [], []
    for ch in chunks:
        v = ch.get("vector")
        if v is None:
            continue
        vectors.append(_as_vec(v))
        meta.append(ch)

    if not vectors:
        raise ValueError("Nessun vettore trovato nei chunk.")

    X = _norm_rows(np.vstack(vectors))  # normalizza per cosine similarity
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)

    p_idx, p_meta = _paths(db_dir)
    faiss.write_index(index, str(p_idx))
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Salvato vettore index in {p_idx} ({len(meta)} chunk)")

# ====== Caricamento DB ======
def load_vectordb(db_dir: str | Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    p_idx, p_meta = _paths(db_dir)
    if not p_idx.exists() or not p_meta.exists():
        raise FileNotFoundError("Database vettoriale non trovato. Crea con build_vectordb().")
    index = faiss.read_index(str(p_idx))
    meta = json.loads(p_meta.read_text(encoding="utf-8"))
    return index, meta

# ====== Ricerca ======
def search_by_vector(
    db_dir: str | Path,
    query_vector: List[float] | np.ndarray,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Ricerca i chunk più simili a un vettore query.
    Restituisce una lista di dict con metadati + 'score' (cosine similarity 0..1).
    """
    index, meta = load_vectordb(db_dir)
    q = _norm_rows(_as_vec(query_vector).reshape(1, -1))
    D, I = index.search(q, top_k)
    D, I = D[0], I[0]

    results = []
    for score, idx in zip(D, I):
        if 0 <= idx < len(meta):
            item = dict(meta[idx])
            item["score"] = float(score)
            results.append(item)
    return results

# # ====== Esempio d'uso ======
# build_vectordb("./vectordb", merged_id)




