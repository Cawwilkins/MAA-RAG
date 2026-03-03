from __future__ import annotations

import os
import re
from typing import List, Optional, Dict, Any
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from concurrent.futures import ThreadPoolExecutor

# Native text extraction
from pypdf import PdfReader

# OCR fallback
import pytesseract
from pdf2image import convert_from_path

from PIL import ImageOps

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
        images = convert_from_path(
            file_path,
            dpi=self._dpi,
            poppler_path=self._poppler_path,
            thread_count=8,
        )

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
                    },
                )
            )
        return docs


poppler_path = r"C:\Users\Christian\Documents\Local_Code\MAA-RAG\Release-25.12.0-0\poppler\Library\bin"


# Feed documents into llama_index, returns list of docs with metadata
def feed_documents(dir_path: str | None = None) -> list[Document]:
    if dir_path is None:
        dir_path = r"C:\Users\Christian\Documents\Local_Code\MAA-RAG\Code\Docs"

    pdf_reader = HybridPDFReader(
        poppler_path=poppler_path,
        min_text_chars=200,   # bump up if your PDFs should have lots of text
        dpi=200,              # 300 is a solid OCR default
        tesseract_lang="eng",
    )

    reader = SimpleDirectoryReader(
        input_dir=dir_path,
        recursive=True,
        exclude_empty=True,
        exclude_hidden=True,
        required_exts=[".pdf", ".PDF"],
        file_extractor={
            ".pdf": pdf_reader,
            ".PDF": pdf_reader,
        },
    )

    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents from {dir_path}")
    return documents


if __name__ == "__main__":
    docs = feed_documents()

    print(f"\nLoaded {len(docs)} documents\n")

    # Inspect a few documents to verify ingestion + metadata
    for d in docs[:5]:
        print("TITLE:", d.metadata.get("title"))
        print("TYPE:", d.metadata.get("doc_type"))
        print("SOURCE:", d.metadata.get("source"))
        print("FILE:", d.metadata.get("file_path"))
        print("PAGE:", d.metadata.get("page"))
        print("TEXT PREVIEW:")
        print(d.text[:300])
        print("-" * 50)
