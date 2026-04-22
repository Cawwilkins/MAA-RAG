import re
from typing import Optional
from config import RATE_ALIASES
from pathlib import Path

from config import EXTRACT_TEXT_WINDOW as TEXT_WINDOW, EXTRACT_JOB_WINDOW as JOB_WINDOW, MAX_VALID_YEAR, MIN_VALID_YEAR

JOB_RE = re.compile(r"\b([A-Z]\d{3,})\b", re.IGNORECASE)

SHIP_RE = re.compile(
    r"\b(USS\s+[A-Z][A-Za-z0-9'’-]*(?:\s+[A-Z][A-Za-z0-9'’-]*){0,4}(?:\s*\([A-Z0-9\-]+\)|\s+[A-Z]{1,3}-\d{1,4})?)\b"
)

SHIP_CLASS_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9'/-]*(?:\s+[A-Z][A-Za-z0-9'/-]*){0,3})\s*[- ]\s*class\b",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
YEAR_RANGE_RE = re.compile(
    r"\b(18\d{2}|19\d{2}|20\d{2})\s*[-–—]\s*(18\d{2}|19\d{2}|20\d{2})\b"
)
FROM_TO_YEAR_RE = re.compile(
    r"\bfrom\s+(18\d{2}|19\d{2}|20\d{2})\s+to\s+(18\d{2}|19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

SHIPYARD_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,5}\s+Naval Shipyard)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,5}\s+Shipyard)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,5}\s+Shipbuilding)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,5}\s+Navy Yard)\b"),
]

DOC_TYPE_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "deck_log",
        [
            re.compile(r"\bdeck logs?\b", re.IGNORECASE),
            re.compile(r"\bdecklogs?\b", re.IGNORECASE),
        ],
    ),
    (
        "milspec",
        [
            re.compile(r"\bmil[- ]?spec\b", re.IGNORECASE),
        ],
    ),
    (
        "cruise_book",
        [
            re.compile(r"\bcruise books?\b", re.IGNORECASE),
            re.compile(r"\bcruisebooks?\b", re.IGNORECASE),
        ],
    ),
    (
        "memorandum",
        [
            re.compile(r"\bmemorandum\b", re.IGNORECASE),
            re.compile(r"\bmemo\b", re.IGNORECASE),
        ],
    ),
    (
        "report",
        [
            re.compile(r"\breport\b", re.IGNORECASE),
        ],
    ),
    (
        "research_document",
        [
            re.compile(r"\bresearch\b", re.IGNORECASE),
        ],
    ),
]


def _build_search_space(title: str, text: str = "", limit: int = TEXT_WINDOW) -> str:
    return f"{title}\n{text[:limit]}" if text else title

def _valid_year(y: int) -> bool:
    return MIN_VALID_YEAR <= y <= MAX_VALID_YEAR

def build_metadata_context(file_path: str | Path, title: str, text: str) -> dict:
    p = Path(file_path)

    return {
        "file_path": str(p),
        "path_obj": p,
        "title": title,
        "title_lower": title.lower(),
        "text": text or "",
        "search_space": _build_search_space(title, text),
        "path_parts_normalized": [_normalize(part) for part in p.parts],
        "title_normalized": _normalize(title),
    }

def extract_job_number(ctx: dict) -> str | None:
    title = ctx["title"]
    text = ctx["text"]
    
    m = JOB_RE.search(title)
    if m:
        return m.group(1).upper()

    m = JOB_RE.search(text[:JOB_WINDOW])
    if m:
        return m.group(1).upper()

    return None


def _normalize(s: str) -> str:
    """Normalize string for matching."""
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)   # deck_logs → deck logs
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def classify_doc_type(ctx: dict) -> str:
    candidates = ctx["path_parts_normalized"] + [ctx["title_normalized"]]

    for doc_type, patterns in DOC_TYPE_PATTERNS:
        for candidate in candidates:
            for pattern in patterns:
                if pattern.search(candidate):
                    return doc_type

    return "undefined_doc"


def extract_ships(ctx: dict) -> tuple[list[str], str | None]:
    search_space = ctx["search_space"]
    title_lower = ctx["title_lower"]

    seen = set()
    ships = []

    for match in SHIP_RE.finditer(search_space):
        ship = " ".join(match.group(1).split()).strip()
        if ship not in seen:
            seen.add(ship)
            ships.append(ship)

    primary_ship = None
    if ships:
        for ship in ships:
            if ship.lower() in title_lower:
                primary_ship = ship
                break
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


def extract_ship_classes(ctx: dict) -> tuple[list[str], str | None]:
    search_space = ctx["search_space"]
    title_lower = ctx["title_lower"]

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
        for sc in ship_classes:
            class_core = sc[:-6].strip().lower()
            if class_core in title_lower:
                primary_ship_class = sc
                break
        if not primary_ship_class:
            primary_ship_class = ship_classes[0]

    return ship_classes, primary_ship_class


def extract_year_info(ctx: dict) -> tuple[list[int], str | None]:
    search_space = ctx["search_space"]

    years_found = set()
    ranges_found: list[tuple[int, int]] = []

    for m in YEAR_RANGE_RE.finditer(search_space):
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        if _valid_year(start) and _valid_year(end):
            ranges_found.append((start, end))
            if end - start <= 25:
                for y in range(start, end + 1):
                    years_found.add(y)
            else:
                years_found.add(start)
                years_found.add(end)

    for m in FROM_TO_YEAR_RE.finditer(search_space):
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        if _valid_year(start) and _valid_year(end):
            ranges_found.append((start, end))
            if end - start <= 25:
                for y in range(start, end + 1):
                    years_found.add(y)
            else:
                years_found.add(start)
                years_found.add(end)

    for m in YEAR_RE.finditer(search_space):
        y = int(m.group(1))
        if _valid_year(y):
            years_found.add(y)
    years_mentioned = sorted(years_found)

    primary_year_range: Optional[str] = None

    if ranges_found:
        start, end = ranges_found[0]
        primary_year_range = f"{start}-{end}"
    elif years_mentioned:
        if len(years_mentioned) == 1:
            primary_year_range = str(years_mentioned[0])
        else:
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

def extract_rates(ctx: dict) -> tuple[list[str], str | None]:
    search_space = ctx["search_space"]
    title_lower = ctx["title_lower"]

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


def _normalize_shipyard(raw: str) -> str:
    return " ".join(raw.split()).strip()

def extract_shipyards(ctx: dict) -> tuple[list[str], str | None]:
    search_space = ctx["search_space"]

    matches: list[tuple[str, int]] = []
    for pattern in SHIPYARD_PATTERNS:
        for m in pattern.finditer(search_space):
            yard = _normalize_shipyard(m.group(1))
            matches.append((yard, m.start()))

    if not matches:
        return [], None

    seen = set()
    shipyards_mentioned = []
    for yard, _pos in sorted(matches, key=lambda x: x[1]):
        if yard not in seen:
            seen.add(yard)
            shipyards_mentioned.append(yard)

    primary_shipyard = shipyards_mentioned[0] if shipyards_mentioned else None
    return shipyards_mentioned, primary_shipyard


def extract_all_metadata(file_path: str | Path, title: str, text: str, source: str | None = None) -> dict:
    ctx = build_metadata_context(file_path, title, text)

    job_num = extract_job_number(ctx)
    doc_type = classify_doc_type(ctx)
    ships, primary_ship = extract_ships(ctx)
    years_mentioned, primary_year_range = extract_year_info(ctx)
    ship_classes, primary_ship_class = extract_ship_classes(ctx)
    rates_mentioned, primary_rate = extract_rates(ctx)
    shipyards_mentioned, primary_shipyard = extract_shipyards(ctx)
    source_quality = classify_source_quality(source, ctx["text"]) if source else None

    return {
        "file_path": ctx["file_path"],
        "title": ctx["title"],
        "job_number": job_num,
        "doc_type": doc_type,
        "ships": ships,
        "primary_ship": primary_ship,
        "years_mentioned": years_mentioned,
        "primary_year_range": primary_year_range,
        "ship_classes": ship_classes,
        "primary_ship_class": primary_ship_class,
        "rates_mentioned": rates_mentioned,
        "primary_rate": primary_rate,
        "shipyards_mentioned": shipyards_mentioned,
        "primary_shipyard": primary_shipyard,
        "source": source,
        "source_quality": source_quality,
    }