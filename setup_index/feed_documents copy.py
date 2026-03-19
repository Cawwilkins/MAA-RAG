from __future__ import annotations
import os
import re
from typing import List, Optional, Dict, Any
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "Docs"

# Converts image to black and white and applies autocontrast to improve OCR accuracy; can be expanded with more preprocessing as needed
def preprocess_for_ocr(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img

# Fix various OCR Issues in one pass, expand as needed
def clean_ocr(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # fix hyphen breaks
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse newlines
    text = re.sub(r"[ \t]{2,}", " ", text)      # extra spaces
    text = re.sub(r"\bPage\s+\d+\b", "", text)  # page numbers
    return text.strip()

# Fix OCR of "I" being read as "|" when it starts a word, in three contexts:
def fix_pipe_pronoun_I(text: str) -> str:
    # 1) Start-of-line: "| have" -> "I have"
    text = re.sub(r'(?m)^\|\s*(?=[A-Za-z])', 'I ', text)

    # 2) After whitespace/punctuation: " . | have" -> " . I have"
    text = re.sub(r'(?<=\s)\|\s*(?=[A-Za-z])', 'I ', text)

    # 3) Standalone pipe at end of sentence/line: "... ). |" -> "... ). I"
    # Only when it looks like a stray OCR char (pipe with optional spaces around it)
    text = re.sub(r'(?<=\S)\s*\|\s*(?=\s|$)', ' I', text)

    return text


# Get ID of documents
def get_source_id(file_path: str | Path, docs_dir: Path) -> str:
    """Stable ID for a source file: relative path from Docs/."""
    return Path(file_path).resolve().relative_to(docs_dir.resolve()).as_posix()

# Get size and last change timestamp of doc
def get_file_state(file_path: str | Path, docs_dir: Path) -> dict:
    """Cheap metadata used to decide whether a file likely changed."""
    path = Path(file_path)
    stat = path.stat()
    return {
        "source_id": get_source_id(path, docs_dir),
        "file_path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class HybridPDFReader(BaseReader):
    """
    Reads PDFs with a hybrid strategy:
    1) Try native PDF text extraction (fast + accurate when PDF has real text)
    2) If extracted text is too small, assume scanned PDF and OCR it (slower)
    Returns one LlamaIndex Document per page with metadata (file_path, page, source).
    """
    def __init__(
        self,
        poppler_path: str,
        min_text_chars: int = 200,
        dpi: int = 200,
        tesseract_lang: str = "eng",
        tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ):
        self._poppler_path = poppler_path
        self._min_text_chars = min_text_chars
        self._dpi = dpi
        self._tesseract_lang = tesseract_lang
        self._tesseract_cmd = tesseract_cmd
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _native_text(self, file_path: str) -> List[str]:
        reader = PdfReader(file_path)
        return [(page.extract_text() or "").strip() for page in reader.pages]

    def _ocr_text(self, file_path: str) -> List[str]:
        kwargs = {
            "dpi": self._dpi,
            "thread_count": 8,
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
        
            # Fix OCR confusion: | -> I when starting a word
            text = fix_pipe_pronoun_I(text)

            return text.strip()
        
        # Threads work well here because pytesseract calls an external process
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(ocr_one, images))


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
        
        source_id = get_source_id(file, DOCS_DIR)
        filename = os.path.basename(file)
        title = os.path.splitext(filename)[0]

        doc_type = ("report" if "report" in title.lower() else "research_document")

        docs: List[Document] = []
        for i, text in enumerate(pages_text, start=1):
            docs.append(
                Document(
                    text=text,
                    metadata={
                        **extra_info,
                        "file_path": str(file), #filepath not json serializable so convert to string
                        "page": i,
                        "source": source,
                        "title": title,
                        "doc_type": doc_type,
                        "source_id": source_id,
                    },
                )
            )
        return docs


def collect_pdf_paths(file_paths: Optional[list[str | Path]] = None) -> list[Path]:
    """
    Normalize and validate a provided list of PDF file paths.
    """
    if not file_paths:
        return []

    cleaned: list[Path] = []
    for p in file_paths:
        path = Path(p).resolve()
        if path.exists() and path.suffix.lower() == ".pdf":
            cleaned.append(path)

    return cleaned



# Feed documents into llama_index, returns list of docs with metadata
def feed_documents(file_paths: Optional[list[str | Path]] = None) -> list[Document]:
    if dir_path is None:
        dir_path = str(DOCS_DIR)

    pdf_reader = HybridPDFReader(
        poppler_path=None,
        min_text_chars=200, 
        dpi=200,              # 300 is a solid OCR default
        tesseract_lang="eng",
    )

    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        recursive=True,
        exclude_empty=True,
        exclude_hidden=True,
        required_exts=[".pdf", ".PDF"],
        file_extractor={
            ".pdf": pdf_reader,
            ".PDF": pdf_reader,
        },
    )
    
    pdf_paths = collect_pdf_paths(file_paths)
    if not pdf_paths:
        return []

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        documents.extend(reader.load_data(pdf_path))

    print(f"Loaded {len(documents)} documents from {dir_path}")
    return documents


if __name__ == "__main__":
    all_pdf_files = list(DOCS_DIR.rglob("*.pdf"))
    docs = feed_documents(file_paths=all_pdf_files)

    print(f"\nLoaded {len(docs)} documents\n")

    for d in docs[:5]:
        print("TITLE:", d.metadata.get("title"))
        print("TYPE:", d.metadata.get("doc_type"))
        print("SOURCE:", d.metadata.get("source"))
        print("FILE:", d.metadata.get("file_path"))
        print("PAGE:", d.metadata.get("page"))
        print("SOURCE_ID:", d.metadata.get("source_id"))
        print("TEXT PREVIEW:")
        print(d.text[:300])
        print("-" * 50)


## IDea is to go through all subfolders, check against json file if that doc is in the db
##if not, write info to json file, then add path to array, this array is what is sent to reader load_data