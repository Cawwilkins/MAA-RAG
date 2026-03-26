from __future__ import annotations
import os
import re
from typing import List, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps
from setup_index.file_utils import get_source_id
#from file_utils import get_source_id

DOCS_DIR = Path(r"C:\Users\Christian.DESKTOP-2DI7LJ6\Documents\Local_Code\MAA-RAG\MAA-RAG Code\Docs")
OCR_THREAD_COUNT = 1
OCR_MAX_WORKERS = 6


def preprocess_for_ocr(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img


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
        title = os.path.splitext(filename)[0]
        doc_type = "report" if "report" in title.lower() else "research_document"

        docs: List[Document] = []
        for i, text in enumerate(pages_text, start=1):
            docs.append(
                Document(
                    text=text,
                    metadata={
                        **extra_info,
                        "file_path": str(Path(file).resolve()),
                        "page": i,
                        "source": source,
                        "title": title,
                        "doc_type": doc_type,
                        "source_id": source_id,
                    },
                )
            )
        return docs


def collect_pdf_paths(file_paths_to_insert: Optional[list[str | Path]] = None) -> list[Path]:
    if not file_paths_to_insert:
        return []

    cleaned: list[Path] = []
    for p in file_paths_to_insert:
        path = Path(p).resolve()
        if path.exists() and path.is_file() and path.suffix.lower() == ".pdf":
            cleaned.append(path)

    return cleaned


def feed_documents(file_paths_to_insert: Optional[list[str | Path]] = None, docs_root: str | Path = DOCS_DIR) -> list[Document]:
    #Took about 56 seconds with 41 docs to read each 
    pdf_paths = collect_pdf_paths(file_paths_to_insert)
    if not pdf_paths:
        print("No valid PDF files provided to feed_documents.")
        return []

    pdf_reader = HybridPDFReader(
        poppler_path=None,
        min_text_chars=200,
        dpi=200,
        tesseract_lang="eng",
        docs_root=docs_root,
    )

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        try:
            documents.extend(pdf_reader.load_data(str(pdf_path)))
        except Exception as e:
            print(f"Failed to load {pdf_path}: {e}")

    print(f"Loaded {len(documents)} document chunks/pages from {len(pdf_paths)} PDF files.")
    return documents


import time

if __name__ == "__main__":
    start = time.time()

    all_pdf_files = list(DOCS_DIR.rglob("*.pdf"))
    docs = feed_documents(file_paths_to_insert=all_pdf_files, docs_root=DOCS_DIR)

    print(f"\nLoaded {len(docs)} documents")
    print(f"Elapsed time: {time.time() - start:.2f} seconds\n")