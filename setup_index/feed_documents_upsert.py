from __future__ import annotations
import os
import re
import subprocess
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps

from setup_index.file_utils import get_source_id
from config import DOCS_DIR, OCR_MAX_WORKERS, OCR_THREAD_COUNT, WPD_PATH
from setup_index.one_pass_metadata_extractor import extract_all_metadata


RichMetadataKey = Tuple[str, int]
RichMetadataMap = Dict[RichMetadataKey, Dict[str, Any]]
SKIP_NAME_KEYWORDS = {"certification", "invoice"}


def preprocess_for_ocr(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img

def is_references_heading(page_text: str) -> bool:
    """
    Returns True if the first non-empty line is a References heading.
    Handles:
    References
    REFERENCES
    References:
    References :
    """
    for line in (page_text or "").splitlines():
        line = line.strip()

        if not line:
            continue

        return bool(re.fullmatch(r"(?i)references\s*:?", line))

    return False


def is_enclosures_heading(page_text: str) -> bool:
    """
    Returns True if the first non-empty line is an Enclosures heading.
    Handles:
    Enclosures
    ENCLOSURES
    Enclosure
    Enclosures:
    Enclosures :
    """
    for line in (page_text or "").splitlines():
        line = line.strip()

        if not line:
            continue

        return bool(re.fullmatch(r"(?i)enclosures?\s*:?", line))

    return False

def clean_ocr(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\bPage\s+\d+\b", "", text)
    return text.strip()


def fix_pipe_pronoun_I(text: str) -> str:
    text = re.sub(r"(?m)^\|\s*(?=[A-Za-z])", "I ", text)
    text = re.sub(r"(?<=\s)\|\s*(?=[A-Za-z])", "I ", text)
    text = re.sub(r"(?<=\S)\s*\|\s*(?=\s|$)", " I", text)
    return text


def clean_plain_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def make_rich_metadata_key(source_id: str, page: int) -> RichMetadataKey:
    return (source_id, page)


def should_skip_ingestion(path: Path) -> bool:
    name = path.stem.lower()
    return any(keyword in name for keyword in SKIP_NAME_KEYWORDS)


def build_minimal_metadata(
    extra_info: Dict[str, Any],
    title: str,
    job_number: Optional[str],
    source_id: str,
    page: int,
) -> Dict[str, Any]:
    """
    Only the compact metadata needed before chunking/splitting.
    """
    return {
        **extra_info,
        "title": title,
        "job_number": job_number,
        "source_id": source_id,
        "page": page,
    }


def build_embed_exclusion_list(full_metadata: Dict[str, Any]) -> List[str]:
    """
    Keep only title + job_number eligible for embedding metadata injection.
    Everything else is excluded from embed text.
    """
    return [
        key for key in full_metadata.keys()
        if key not in {"title", "job_number"}
    ]


def build_rich_metadata_for_storage(
    metadata_fields: Dict[str, Any],
    extra_info: Dict[str, Any],
    source_id: str,
    page: int,
) -> Dict[str, Any]:
    """
    Full metadata stored outside the Document until after chunking.
    """
    return {
        **extra_info,
        **metadata_fields,
        "source_id": source_id,
        "page": page,
    }


class HybridPDFReader(BaseReader):
    def __init__(
        self,
        poppler_path: Optional[str] = None,
        min_text_chars: int = 200,
        dpi: int = 200,
        tesseract_lang: str = "eng",
        tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        docs_root: str | Path = DOCS_DIR,
    ):
        self._poppler_path = poppler_path
        self._min_text_chars = min_text_chars
        self._dpi = dpi
        self._tesseract_lang = tesseract_lang
        self._tesseract_cmd = tesseract_cmd
        self._docs_root = Path(docs_root).resolve()
        self._rich_metadata_map: RichMetadataMap = {}

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _native_text(self, file_path: str) -> List[str]:
        reader = PdfReader(file_path)
        return [(page.extract_text() or "").strip() for page in reader.pages]

    def _ocr_text(self, file_path: str) -> List[str]:
        kwargs = {
            "dpi": self._dpi,
            "thread_count": OCR_THREAD_COUNT,
        }

        if self._poppler_path:
            kwargs["poppler_path"] = self._poppler_path

        images = convert_from_path(file_path, **kwargs)

        def ocr_one(img):
            img = preprocess_for_ocr(img)
            text = pytesseract.image_to_string(
                img,
                lang=self._tesseract_lang,
                config="--oem 1 --psm 6"
            )
            text = fix_pipe_pronoun_I(text)
            return text.strip()

        with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as ex:
            return list(ex.map(ocr_one, images))

    def get_rich_metadata_map(self) -> RichMetadataMap:
        return dict(self._rich_metadata_map)

    def clear_rich_metadata_map(self) -> None:
        self._rich_metadata_map.clear()

    def load_data(
        self,
        file: str,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        extra_info = extra_info or {}

        native_pages = self._native_text(file)
        total_chars = sum(len(t) for t in native_pages)

        if total_chars >= self._min_text_chars:
            pages_text = native_pages
            source = "pdf_text"
        else:
            pages_text = self._ocr_text(file)
            pages_text = [clean_ocr(t) for t in pages_text]
            source = "ocr_pdf"

        source_id = get_source_id(file, self._docs_root)
        filename = os.path.basename(file)
        file_path = str(Path(file).resolve())
        title = os.path.splitext(filename)[0]

        combined_text = "\n".join(pages_text)
        metadata_fields = extract_all_metadata(file_path, title, combined_text, source)
        doc_type = str(metadata_fields.get("doc_type", "")).lower()
        in_references_section = False
        docs: List[Document] = []

        for i, page_text in enumerate(pages_text, start=1):
            if (
                doc_type == "report" and is_references_heading(page_text)
            ) or (
                doc_type == "memorandum" and is_enclosures_heading(page_text)
            ):
                in_references_section = True

            page_section_metadata = {}
            if in_references_section:
                page_section_metadata = {
                    "section": "references",
                    "is_reference_page": True,
                    "page_role": "reference_page",
                }
            # Store rich metadata externally for later reattachment.
            rich_metadata = build_rich_metadata_for_storage(
                metadata_fields=metadata_fields,
                extra_info=extra_info,
                source_id=source_id,
                page=i,
            )
            self._rich_metadata_map[make_rich_metadata_key(source_id, i)] = rich_metadata

            # Only attach compact metadata before pipeline chunking.
            minimal_metadata = build_minimal_metadata(
                extra_info={**extra_info, **page_section_metadata},
                title=title,
                job_number=metadata_fields.get("job_number"),
                source_id=source_id,
                page=i,
            )

            exclude_from_embed = build_embed_exclusion_list(minimal_metadata)

            docs.append(
                Document(
                    text=page_text,
                    metadata=minimal_metadata,
                    excluded_embed_metadata_keys=exclude_from_embed,
                    excluded_llm_metadata_keys=[],
                )
            )

        return docs


class WPDReader(BaseReader):
    def __init__(
        self,
        wpd2text_path: str | Path = WPD_PATH,
        docs_root: str | Path = DOCS_DIR,
    ):
        self._wpd2text_path = Path(wpd2text_path)
        self._docs_root = Path(docs_root).resolve()
        self._rich_metadata_map: RichMetadataMap = {}

    def _extract_text(self, file_path: str) -> str:
        cmd = [str(self._wpd2text_path), str(Path(file_path).resolve())]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"wpd2text failed for {file_path}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return result.stdout.strip()

    def get_rich_metadata_map(self) -> RichMetadataMap:
        return dict(self._rich_metadata_map)

    def clear_rich_metadata_map(self) -> None:
        self._rich_metadata_map.clear()

    def load_data(
        self,
        file: str,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        extra_info = extra_info or {}

        text = self._extract_text(file)
        text = clean_plain_text(text)

        source_id = get_source_id(file, self._docs_root)
        filename = os.path.basename(file)
        title = os.path.splitext(filename)[0]
        file_path = str(Path(file).resolve())
        source = "wpd_text"
        page = 1

        metadata_fields = extract_all_metadata(file_path, title, text, source)
        doc_type = str(metadata_fields.get("doc_type", "")).lower()
        section_metadata = {}

        if (
            doc_type == "report" and is_references_heading(text)
        ) or (
            doc_type == "memorandum" and is_enclosures_heading(text)
        ):
            section_metadata = {
                "section": "references",
                "is_reference_page": True,
                "page_role": "reference_page",
            }

        rich_metadata = build_rich_metadata_for_storage(
            metadata_fields=metadata_fields,
            extra_info=extra_info,
            source_id=source_id,
            page=page,
        )
        self._rich_metadata_map[make_rich_metadata_key(source_id, page)] = rich_metadata

        minimal_metadata = build_minimal_metadata(
            extra_info={**extra_info, **section_metadata},
            title=title,
            job_number=metadata_fields.get("job_number"),
            source_id=source_id,
            page=page,
        )

        exclude_from_embed = build_embed_exclusion_list(minimal_metadata)

        return [
            Document(
                text=text,
                metadata=minimal_metadata,
                excluded_embed_metadata_keys=exclude_from_embed,
                excluded_llm_metadata_keys=[],
            )
        ]


def reattach_rich_metadata_to_nodes(
    nodes,
    rich_metadata_map,
    overwrite=False,
):
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}

        source_id = metadata.get("source_id")
        page = metadata.get("page")

        if source_id is None or page is None:
            continue

        rich = rich_metadata_map.get((source_id, int(page)))
        if not rich:
            continue

        for key, value in rich.items():
            if overwrite or key not in metadata:
                metadata[key] = value

        existing_embed_excluded = set(
            getattr(node, "excluded_embed_metadata_keys", []) or []
        )
        existing_embed_excluded.update(rich.keys())
        node.excluded_embed_metadata_keys = list(existing_embed_excluded)

        existing_llm_excluded = set(
            getattr(node, "excluded_llm_metadata_keys", []) or []
        )
        existing_llm_excluded.update(rich.keys())
        node.excluded_llm_metadata_keys = list(existing_llm_excluded)

    return nodes


def collect_file_paths(file_paths_to_insert: Optional[list[str | Path]] = None) -> list[Path]:
    if not file_paths_to_insert:
        return []

    cleaned: list[Path] = []
    for p in file_paths_to_insert:
        path = Path(p).resolve()
        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in {".pdf", ".wpd"}
            and not should_skip_ingestion(path)
        ):
            cleaned.append(path)

    return cleaned


def feed_documents(
    file_paths_to_insert: Optional[list[str | Path]] = None,
    docs_root: str | Path = DOCS_DIR
) -> tuple[list[Document], RichMetadataMap]:
    paths = collect_file_paths(file_paths_to_insert)
    if not paths:
        print("No valid PDF or WPD files provided to feed_documents.")
        return [], {}

    pdf_reader = HybridPDFReader(
        poppler_path=None,
        min_text_chars=200,
        dpi=200,
        tesseract_lang="eng",
        docs_root=docs_root,
    )

    wpd_reader = WPDReader(
        wpd2text_path=WPD_PATH,
        docs_root=docs_root,
    )

    documents: list[Document] = []
    rich_metadata_map: RichMetadataMap = {}

    for path in paths:
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                docs = pdf_reader.load_data(str(path))
                documents.extend(docs)
                rich_metadata_map.update(pdf_reader.get_rich_metadata_map())
                pdf_reader.clear_rich_metadata_map()

            elif suffix == ".wpd":
                docs = wpd_reader.load_data(str(path))
                documents.extend(docs)
                rich_metadata_map.update(wpd_reader.get_rich_metadata_map())
                wpd_reader.clear_rich_metadata_map()

        except Exception as e:
            print(f"Failed to load {path}: {e}")

    print(f"Loaded {len(documents)} documents from {len(paths)} PDF/WPD files.")
    return documents, rich_metadata_map