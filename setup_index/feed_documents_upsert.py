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
import subprocess
from setup_index.file_utils import get_source_id
from config import DOCS_DIR, OCR_MAX_WORKERS, OCR_THREAD_COUNT, WPD_PATH, RATE_ALIASES


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


def extract_job_number(title: str, text: str = "") -> str | None:
    JOB_RE = re.compile(r"\b([A-Z]\d{3,})\b", re.IGNORECASE)

    m = JOB_RE.search(title)
    if m:
        return m.group(1).upper()

    m = JOB_RE.search(text[:2000])
    if m:
        return m.group(1).upper()

    return None


def _normalize(s: str) -> str:
    """Normalize string for matching."""
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)   # deck_logs → deck logs
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def classify_doc_type(file_path: str | Path, title: str) -> str:
    p = Path(file_path)

    # Normalize path parts and title
    parts = [_normalize(part) for part in p.parts]
    title_str = _normalize(title)

    # Combine for matching (but still preserve structure)
    all_strings = parts + [title_str]

    def contains_any(keywords: list[str]) -> bool:
        for kw in keywords:
            kw = _normalize(kw)
            for s in all_strings:
                if kw in s:
                    return True
        return False

    # --- Priority order matters ---
    # Most specific → most general

    if contains_any(["deck log", "deck logs", "decklog", "decklogs"]):
        return "deck_log"

    if contains_any(["milspec", "mil spec", "mil-spec"]):
        return "milspec"

    if contains_any(["cruise book", "cruise books", "cruisebook", "cruisebooks"]):
        return "cruise_book"

    if contains_any(["memorandum", "memo"]):
        return "memorandum"

    if contains_any(["report"]):
        return "report"

    if contains_any(["research"]):
        return "research_document"

    return "undefined_doc"


def extract_ships(title: str, text: str = "") -> tuple[list[str], str | None]:
    SHIP_RE = re.compile(
        r"\b(USS\s+[A-Z][A-Za-z0-9\-']+(?:\s+[A-Z][A-Za-z0-9\-']+){0,3})\b"
    )
    
    search_space = f"{title}\n{text[:5000]}"

    seen = set()
    ships = []

    # Extract all unique ships (order preserved)
    for match in SHIP_RE.finditer(search_space):
        ship = match.group(1).strip()
        if ship not in seen:
            seen.add(ship)
            ships.append(ship)

    # --- Determine primary ship ---
    primary_ship = None

    if ships:
        title_lower = title.lower()

        # 1. If a ship appears in title → strongest signal
        for ship in ships:
            if ship.lower() in title_lower:
                primary_ship = ship
                break

        # 2. Otherwise fallback to first occurrence
        if not primary_ship:
            primary_ship = ships[0]

    return ships, primary_ship


def _normalize_ship_class(raw: str) -> str:
    raw = raw.strip()

    # normalize separators/spaces
    raw = re.sub(r"[_\s]+", " ", raw)
    raw = raw.replace("–", "-").replace("—", "-")

    # title-case each space-separated token, preserve internal hyphens
    parts = []
    for token in raw.split():
        subparts = token.split("-")
        subparts = [sp.capitalize() for sp in subparts if sp]
        parts.append("-".join(subparts))

    return " ".join(parts) + " Class"


def extract_ship_classes(title: str, text: str = "") -> tuple[list[str], str | None]:
    SHIP_CLASS_RE = re.compile(
        r"\b([A-Z][A-Za-z0-9'/-]{1,40})\s*[- ]\s*class\b",
        re.IGNORECASE,
    )
    search_space = f"{title}\n{text[:5000]}"

    seen = set()
    ship_classes = []

    for match in SHIP_CLASS_RE.finditer(search_space):
        raw_class = match.group(1)
        normalized = _normalize_ship_class(raw_class)

        if normalized not in seen:
            seen.add(normalized)
            ship_classes.append(normalized)

    primary_ship_class = None

    if ship_classes:
        title_lower = title.lower()

        # strongest signal: class appears in title
        for sc in ship_classes:
            class_core = sc[:-6].strip().lower() if sc.lower().endswith(" class") else sc.lower()
            if class_core in title_lower:
                primary_ship_class = sc
                break

        # fallback: first explicit mention
        if not primary_ship_class:
            primary_ship_class = ship_classes[0]

    return ship_classes, primary_ship_class


def extract_year_info(title: str, text: str = "") -> tuple[list[int], str | None]:
    YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
    YEAR_RANGE_RE = re.compile(
        r"\b(18\d{2}|19\d{2}|20\d{2})\s*[-–—]\s*(18\d{2}|19\d{2}|20\d{2})\b"
    )
    FROM_TO_YEAR_RE = re.compile(
        r"\bfrom\s+(18\d{2}|19\d{2}|20\d{2})\s+to\s+(18\d{2}|19\d{2}|20\d{2})\b",
        re.IGNORECASE,
    )
    """
    Returns:
        years_mentioned: sorted unique years
        primary_year_range: best single range/string for header display
    """
    search_space = f"{title}\n{text[:5000]}"

    years_found = set()
    ranges_found: list[tuple[int, int]] = []

    # 1. Explicit ranges like 1943-1945
    for m in YEAR_RANGE_RE.finditer(search_space):
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        ranges_found.append((start, end))
        for y in range(start, end + 1):
            years_found.add(y)

    # 2. "from 1943 to 1945"
    for m in FROM_TO_YEAR_RE.finditer(search_space):
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        ranges_found.append((start, end))
        for y in range(start, end + 1):
            years_found.add(y)

    # 3. Single years
    for m in YEAR_RE.finditer(search_space):
        years_found.add(int(m.group(1)))

    years_mentioned = sorted(years_found)

    # Pick a primary range
    primary_year_range: Optional[str] = None

    if ranges_found:
        # choose the first explicit range found in text/title
        start, end = ranges_found[0]
        primary_year_range = f"{start}-{end}"
    elif years_mentioned:
        if len(years_mentioned) == 1:
            primary_year_range = str(years_mentioned[0])
        else:
            # compact fallback: min-max of extracted years
            primary_year_range = f"{years_mentioned[0]}-{years_mentioned[-1]}"

    return years_mentioned, primary_year_range


def classify_source_quality(source: Optional[str], text: str) -> str | None:
    """
    Classify document text quality for retrieval/debugging.

    Returns:
        - "digital"    : clean native text extraction
        - "ocr_clean"  : OCR text but reasonably readable
        - "ocr_noisy"  : OCR text with obvious corruption
        - None         : unknown
    """
    if not source:
        return None

    source = source.lower()
    sample = (text or "")[:3000]

    if source in {"pdf_text", "wpd_text"}:
        return "digital"

    if source != "ocr_pdf":
        return None

    # OCR quality heuristics
    if not sample.strip():
        return "ocr_noisy"

    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    words = re.findall(r"\b\w+\b", sample)
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]

    total_words = len(words)
    if total_words == 0:
        return "ocr_noisy"

    # suspicious chars often seen in bad OCR
    weird_char_count = len(re.findall(r"[|~`^_]{1,}", sample))

    # words with lots of non-alpha junk
    junky_words = 0
    for w in alpha_words:
        non_alpha = sum(1 for ch in w if not ch.isalpha())
        if len(w) >= 4 and non_alpha / len(w) > 0.35:
            junky_words += 1

    junk_ratio = junky_words / max(len(alpha_words), 1)

    # very short fragmented lines can indicate messy OCR
    short_lines = sum(1 for line in lines if len(line) < 20)
    short_line_ratio = short_lines / max(len(lines), 1)

    # decide quality
    if weird_char_count > 20 or junk_ratio > 0.18 or short_line_ratio > 0.75:
        return "ocr_noisy"

    return "ocr_clean"


def build_rate_patterns(rate_aliases: dict[str, list[str]]):
    compiled = []

    for canonical, aliases in rate_aliases.items():
        for alias in aliases:
            compiled.append(
                (
                    canonical,
                    alias,
                    re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                )
            )

    return compiled

RATE_PATTERNS = build_rate_patterns(RATE_ALIASES)

def extract_rates(title: str, text: str = "") -> tuple[list[str], str | None]:
    search_space = f"{title}\n{text[:5000]}"

    found_positions = {}
    rates_mentioned = []

    # Find all unique canonical matches and first positions
    for canonical, alias, pattern in RATE_PATTERNS:
        match = pattern.search(search_space)
        if match:
            if canonical not in found_positions or match.start() < found_positions[canonical]:
                found_positions[canonical] = match.start()

    # Sort canonicals by first appearance in search space
    for canonical, _pos in sorted(found_positions.items(), key=lambda item: item[1]):
        rates_mentioned.append(canonical)

    primary_rate = None

    if rates_mentioned:
        title_lower = title.lower()

        # 1. strongest signal: appears in title
        for canonical, aliases in RATE_ALIASES.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", title_lower, re.IGNORECASE):
                    if canonical in rates_mentioned:
                        primary_rate = canonical
                        break
            if primary_rate:
                break

        # 2. fallback: first mention in title/text
        if not primary_rate:
            primary_rate = rates_mentioned[0]

    return rates_mentioned, primary_rate

SHIPYARD_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-z&.\-'\s]{2,60} Naval Shipyard)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.\-'\s]{2,60} Shipyard)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.\-'\s]{2,60} Shipbuilding)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.\-'\s]{2,60} Navy Yard)\b"),
]

def _normalize_shipyard(raw: str) -> str:
    return " ".join(raw.split()).strip()

def extract_shipyards(title: str, text: str = "") -> tuple[list[str], str | None]:
    search_space = f"{title}\n{text[:5000]}"

    matches: list[tuple[str, int]] = []

    for pattern in SHIPYARD_PATTERNS:
        for m in pattern.finditer(search_space):
            yard = _normalize_shipyard(m.group(1))
            matches.append((yard, m.start()))

    if not matches:
        return [], None

    # dedupe while preserving earliest occurrence
    seen = set()
    shipyards_mentioned = []
    for yard, _pos in sorted(matches, key=lambda x: x[1]):
        if yard not in seen:
            seen.add(yard)
            shipyards_mentioned.append(yard)

    primary_shipyard = shipyards_mentioned[0] if shipyards_mentioned else None
    return shipyards_mentioned, primary_shipyard

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

        combined_text = "\n".join(pages_text[:5])
        job_num = extract_job_number(title, combined_text)
        
        full_text = "\n".join(pages_text[:5])
        source_quality = classify_source_quality(source, full_text)

        doc_type = classify_doc_type(file, title)
        ships, primary_ship = extract_ships(title, combined_text)
        
        combined_text = "\n".join(pages_text[:5])
        years_mentioned, primary_year_range = extract_year_info(title, combined_text)

        combined_text = "\n".join(pages_text[:5])
        ship_classes, primary_ship_class = extract_ship_classes(title, combined_text)

        combined_text = "\n".join(pages_text[:5])
        rates_mentioned, primary_rate = extract_rates(title, combined_text)
        shipyards_mentioned, primary_shipyard = extract_shipyards(title, combined_text)

        docs: List[Document] = []
        for i, page_text in enumerate(pages_text, start=1):
            docs.append(
                Document(
                    text=page_text,
                    metadata={
                        **extra_info,
                        "file_path": str(Path(file).resolve()),
                        "page": i,
                        "source": source,
                        "title": title,
                        "ships": ships,
                        "primary_ship": primary_ship,
                        "source_quality": source_quality,
                        "job_number": job_num,
                        "doc_type": doc_type,
                        "source_id": source_id,
                        "years_mentioned": years_mentioned,
                        "primary_year_range": primary_year_range,
                        "ship_classes": ship_classes,
                        "primary_ship_class": primary_ship_class,
                        "rates_mentioned": rates_mentioned,
                        "primary_rate": primary_rate,
                        "shipyards_mentioned": shipyards_mentioned,
                        "primary_shipyard": primary_shipyard
                    },
                )
            )
        return docs


def clean_plain_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

class WPDReader(BaseReader):
    def __init__(
        self,
        wpd2text_path: str | Path = WPD_PATH,
        docs_root: str | Path = DOCS_DIR,
    ):
        self._wpd2text_path = Path(wpd2text_path)
        self._docs_root = Path(docs_root).resolve()

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
        job_num = extract_job_number(title, text)
        doc_type = classify_doc_type(file, title)
        ships, primary_ship = extract_ships(title, text)
        source_quality = classify_source_quality("wpd_text", text)
        years_mentioned, primary_year_range = extract_year_info(title, text)
        ship_classes, primary_ship_class = extract_ship_classes(title, text)
        rates_mentioned, primary_rate = extract_rates(title, text)
        shipyards_mentioned, primary_shipyard = extract_shipyards(title, text)

        return [
            Document(
                text=text,
                metadata={
                    **extra_info,
                    "file_path": str(Path(file).resolve()),
                    "page": 1,
                    "source": "wpd_text",
                    "title": title,
                    "ships": ships,
                    "primary_ship": primary_ship,
                    "job_number":job_num,
                    "source_quality": source_quality,
                    "doc_type": doc_type,
                    "source_id": source_id,
                    "years_mentioned": years_mentioned,
                    "primary_year_range": primary_year_range,
                    "ship_classes": ship_classes,
                    "primary_ship_class": primary_ship_class,
                    "rates_mentioned": rates_mentioned,
                    "primary_rate": primary_rate,
                    "shipyards_mentioned": shipyards_mentioned,
                    "primary_shipyard": primary_shipyard,
                },
            )
        ]

def collect_file_paths(file_paths_to_insert: Optional[list[str | Path]] = None) -> list[Path]:
    if not file_paths_to_insert:
        return []

    cleaned: list[Path] = []
    for p in file_paths_to_insert:
        path = Path(p).resolve()
        if path.exists() and path.is_file() and path.suffix.lower() in {".pdf", ".wpd"}:
            cleaned.append(path)

    return cleaned


def feed_documents(file_paths_to_insert: Optional[list[str | Path]] = None, docs_root: str | Path = DOCS_DIR) -> list[Document]:
    #Took about 56 seconds with 41 docs to read each 
    paths = collect_file_paths(file_paths_to_insert)
    if not paths:
        print("No valid PDF or WPD files provided to feed_documents.")
        return []

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

    for path in paths:
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                documents.extend(pdf_reader.load_data(str(path)))
            elif suffix == ".wpd":
                documents.extend(wpd_reader.load_data(str(path)))
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    print(f"Loaded {len(documents)} documents from {len(paths)} PDF/WPD files.")
    return documents


import time

if __name__ == "__main__":
    start = time.time()

    all_files = [
        p for p in DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".wpd"}
    ]
    docs = feed_documents(file_paths_to_insert=all_files, docs_root=DOCS_DIR)   

    print(f"\nLoaded {len(docs)} documents")
    print(f"Elapsed time: {time.time() - start:.2f} seconds\n")