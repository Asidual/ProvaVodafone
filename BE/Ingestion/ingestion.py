# pipeline_debug.py
# ------------------------------------------------------------
# Pipeline RAG con debug esteso e log a livello DEBUG.
# Mantiene i commenti originali e aggiunge log granulari per ogni step.
# ------------------------------------------------------------

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv  # type: ignore
load_dotenv()

# == LOGGING ==
logging.basicConfig(
    level=logging.DEBUG,  # <-- imposta DEBUG
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("pipeline")

# == UTILS IMPORT ==
from utils import pdf_to_markdown
from utils import normalize_numeric_headings, demote_unnumbered_headers_to_bold
from utils import (
    split_markdown_build_records_by_page_markers_no_recursive,
    correct_fig_assigment,
    add_embedding,
)
from utils import (
    extract_figures_png_robust_from_dict,
    build_fig_items_from_records,
    describe_from_items,
    add_embedding_image,
)
from utils import merge_text_and_image_records, add_chunk_ids
from utils import build_vectordb


# ---------------------------------------------
# Config “semplice” per file/path di lavoro
# ---------------------------------------------
URL_PDF = r"BE\Ingestion\Documents\printf-manuale.pdf"

# GIA PRODOTTO SENZA DOVER FARE IL RUN DEL DOC INTELLIGENCE
MD_IN_PATH   = Path(r"BE\Ingestion\Documents\old_json\1_print_page4.md")
DI_JSON_PATH = Path(r"BE\Ingestion\Documents\old_json\1_result_info.json")

# Output (se abilitiamo salvataggi intermedi)
DEBUG_OUT_DIR = Path(r"BE\Ingestion\Documents\debug_runs")  # cartella per salvataggi intermedi
DEBUG_SAVE_INTERMEDIATE = True  # setta a False per non salvare

# Directory figure di output
FIGURES_DIR = r"BE\Ingestion\figures2"

# Directory finale del vector db
VECTORDIR_OUT = Path(r"BE\AllVectorDB\vectordb2")


def _timed(step_name: str):
    """Decorator semplice per loggare durata step."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            logger.debug(f"▶️  START: {step_name}")
            t0 = time.time()
            try:
                res = fn(*args, **kwargs)
                dt = (time.time() - t0) * 1000
                logger.debug(f"✅ END: {step_name} — {dt:.0f} ms")
                return res
            except Exception as e:
                logger.exception(f"❌ ERROR in {step_name}: {e}")
                raise
        return wrapper
    return deco


def _save_debug(obj: Any, out_path: Path):
    """Salva dict/list/testo in JSON/MD per debug."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, (dict, list)):
            out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        elif isinstance(obj, str):
            out_path.write_text(obj, encoding="utf-8")
        else:
            out_path.write_text(repr(obj), encoding="utf-8")
        logger.debug(f"💾 Salvato debug: {out_path}")
    except Exception:
        logger.warning(f"Impossibile salvare {out_path}", exc_info=True)


@_timed("LOAD INPUT MARKDOWN & DI POOLER")
def load_inputs(md_path: Path, di_path: Path) -> tuple[str, Dict[str, Any]]:
    # GIA PRODOTTO SENZA DOVER FARE IL RUN DEL DOC INTELLIGENCE
    with open(md_path, "r", encoding="utf8") as f:
        markdown = f.read()  # Il markdown grezzo generato da Document Intelligente
        print("Importato markdown")
        logger.info(f"Importato markdown da: {md_path}")

    with open(di_path, "r", encoding="utf8") as f:
        di_pooler = json.load(f)  # I dati sulle figure e il loro posizionamento all'interno del pdf
        print("Importato markdown")
        logger.info(f"Importato info Document Intelligence da: {di_path}")

    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(markdown, DEBUG_OUT_DIR / "1_markdown_original.md")
        _save_debug(di_pooler, DEBUG_OUT_DIR / "1_result_info.json")

    return markdown, di_pooler


@_timed("NORMALIZE HEADINGS (1/2)")
def normalize_h1(markdown: str) -> str:
    # Faccio in modo da avere solo # ## ### i livelli dei titoli fino a 3
    markdown = normalize_numeric_headings(markdown, 1, 3)
    print("Normalizzato 1")
    logger.info("Normalizzato headings numerici a max livello 3 (#, ##, ###)")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(markdown, DEBUG_OUT_DIR / "2_markdown_norm_head.md")
    return markdown


@_timed("NORMALIZE HEADINGS (2/2)")
def normalize_h2(markdown: str) -> str:
    markdown = demote_unnumbered_headers_to_bold(markdown)
    print("Normalizzato 2")
    logger.info("Declassati header non numerati a **grassetto**")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(markdown, DEBUG_OUT_DIR / "3_markdown_norm_head_no_sup_head.md")
    return markdown


@_timed("EXTRACT FIGURES")
def extract_figures(url_pdf: str, di_pooler: Dict[str, Any], out_dir: str) -> List[Dict[str, Any]]:
    # Estraggo le immagini e me le salvo in una cartella per l'utilizza a FE con il loro path
    figure_list = extract_figures_png_robust_from_dict(url_pdf, di_pooler, out_dir)
    logger.info(f"Estratte figure: {len(figure_list)}")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(figure_list, DEBUG_OUT_DIR / "4_list_fig.json")
    return figure_list


@_timed("SPLIT & BUILD RECORDS")
def build_records(markdown: str, figure_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Divido basato sul Titolo senza fare il recursive. I testi non sono lunghi e manteniamo compattezza dell'argomento senza dover dividere il testo
    records = split_markdown_build_records_by_page_markers_no_recursive(
        md_text=markdown,                # markdown DI con <!-- PageNumber="N" -->
        figs=figure_list,                # [(path, page), ...]
        headers_to_split_on=(("#","chapter"),("##","section"),("###","subsection")),
        strip_headers=False,
        figures_link_base=FIGURES_DIR,
    )
    logger.info(f"Creati records: {len(records)}")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(records, DEBUG_OUT_DIR / "5_records_split.json")
    return records


@_timed("FIX FIG-ASSIGNMENT")
def fix_fig_assignment(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = correct_fig_assigment(records)  # Assegniamo correttamente le immagini al capitolo o sezione di riferimento
    logger.info("Riassegnazione figure completata")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(records, DEBUG_OUT_DIR / "6_records_fig_fixed.json")
    return records


@_timed("EMBEDDING (TEXT)")
def embed_text(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records_emd_official = add_embedding(records)  # aggiungiamo gli embeddings
    logger.info("Embedding testuali aggiunti")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(records_emd_official, DEBUG_OUT_DIR / "7_records_text_embedd.json")
    return records_emd_official


@_timed("BUILD FIG ITEMS")
def build_fig_items(records_emd_official: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # creiamo i metadati delle figure
    fig_items = build_fig_items_from_records(
        records_emd_official,
        figures_link_base=FIGURES_DIR
    )
    logger.info(f"Figure derivati da records: {len(fig_items)}")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(fig_items, DEBUG_OUT_DIR / "8_fig_items.json")
    return fig_items


@_timed("DESCRIBE FIGS (GPT)")
def describe_figs(fig_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Aggiungiamo la descrizione basato sul modello gpt-4o per aggiungere cosa rappresenta la figura
    img_records = describe_from_items(fig_items)
    logger.info(f"Descrizioni immagini create: {len(img_records)}")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(img_records, DEBUG_OUT_DIR / "9_image_list_description.json")
    return img_records


@_timed("EMBEDDING (IMAGES DESCRIPTIONS)")
def embed_image_desc(img_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # embeddiamo il testo di cosa parla la figura
    results_image_embedd = add_embedding_image(img_records)
    logger.info("Embedding descrizioni immagini aggiunti")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(results_image_embedd, DEBUG_OUT_DIR / "10_image_description_embedd.json")
    return results_image_embedd


@_timed("MERGE TEXT + IMAGES")
def merge_records(records_emd_official: List[Dict[str, Any]], results_image_embedd: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Facciamo il merging sia dei dati della figura che del testo in un unica lista
    merged = merge_text_and_image_records(records_emd_official, results_image_embedd)
    logger.info(f"Merged items: {len(merged)}")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(merged, DEBUG_OUT_DIR / "11_merged_text_images.json")
    return merged


@_timed("ASSIGN CHUNK IDS")
def assign_chunk_ids(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # creiamo degli id univoci
    merged_id = add_chunk_ids(merged)
    logger.info("Chunk IDs assegnati")
    if DEBUG_SAVE_INTERMEDIATE:
        _save_debug(merged_id, DEBUG_OUT_DIR / "12_merged_with_ids.json")
    return merged_id


@_timed("BUILD VECTOR DB (FAISS)")
def build_vector_db(out_dir: Path, merged_id: List[Dict[str, Any]]) -> None:
    # costruiamo il faiss
    build_vectordb(str(out_dir), merged_id)
    logger.info(f"Vector DB creato in: {out_dir.resolve()}")


def main():
    logger.info("=== Pipeline RAG — RUN (DEBUG) ===")
    logger.debug(f"PDF sorgente: {URL_PDF}")
    logger.debug(f"Markdown input path: {MD_IN_PATH}")
    logger.debug(f"DI json path: {DI_JSON_PATH}")
    logger.debug(f"Figures dir (output): {FIGURES_DIR}")
    logger.debug(f"VectorDB out dir: {VECTORDIR_OUT}")

    #STEP 1: Creare il markdown
    # url_manuale = "printf-manuale.pdf"
    # markdown, result = pdf_to_markdown(
    #     file_path=url_manuale,
    #     endpoint=END_DOC,
    #     api_key=KEY_DOC,
    #     model_id="prebuilt-layout",
    #     starting_page=4,  # da quale pagina iniziare (1-based)
    # )
    #
    # GIA PRODOTTO SENZA DOVER FARE IL RUN DEL DOC INTELLIGENCE

    # LOAD INPUTS
    markdown, di_pooler = load_inputs(MD_IN_PATH, DI_JSON_PATH)

    # NORMALIZE HEADERS
    markdown = normalize_h1(markdown)  # Faccio in modo da avere solo # ## ### i livelli dei titoli fino a 3
    markdown = normalize_h2(markdown)  # Faccio il match con il titolo in formato Esempio 1 , 1.2, 1.2.2 ...

    # FIGURES
    figure_list = extract_figures(URL_PDF, di_pooler, FIGURES_DIR)  # Estraggo le immagini e me le salvo in una cartella per FE

    # SPLIT + RECORDS
    records = build_records(markdown, figure_list)  # Divido basato sul Titolo (no recursive, testi compatti)
    records = fix_fig_assignment(records)           # Assegniamo correttamente le immagini al capitolo o sezione di riferimento

    # EMBEDDINGS (TESTO)
    records_emd_official = embed_text(records)      # aggiungiamo gli embeddings

    # -- Images
    fig_items = build_fig_items(records_emd_official)       # creiamo i metadati delle figure
    img_records = describe_figs(fig_items)                  # Aggiungiamo la descrizione basata su gpt-4o
    results_image_embedd = embed_image_desc(img_records)    # embeddiamo il testo di cosa parla la figura

    # -- Merging
    merged = merge_records(records_emd_official, results_image_embedd)  # Facciamo il merging
    merged_id = assign_chunk_ids(merged)                                # creiamo degli id univoci

    # VECTOR DB
    build_vector_db(VECTORDIR_OUT, merged_id)  # costruiamo il faiss

    logger.info("=== Pipeline COMPLETATA con successo ===")


if __name__ == "__main__":
    main()
