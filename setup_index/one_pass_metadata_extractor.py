import re
from pathlib import Path
from typing import Optional

from config import RATE_ALIASES
from config import (
    EXTRACT_TEXT_WINDOW as TEXT_WINDOW,
    EXTRACT_JOB_WINDOW as JOB_WINDOW,
    MAX_VALID_YEAR,
    MIN_VALID_YEAR,
)

JOB_RE = re.compile(r"\b([A-Z]\d{3,})\b", re.IGNORECASE)

SHIP_RE = re.compile(
    r"\b(USS\s+[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,2})\b"
)

SHIP_CLASS_RE = re.compile(
    r"\b(USS\s+[A-Z][A-Za-z0-9'.&/-]*(?:\s+[A-Z][A-Za-z0-9'.&/-]*){0,4})-class\b",
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

SHIPYARD_RE = re.compile(
    r"\b([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*){0,5}\s+"
    r"(?:Naval Shipyard|Shipyard|Shipbuilding|Navy Yard))\b"
)

DOC_TYPE_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "qpl",
        [
            re.compile(r"\bqpl(?:\b|[-\s])", re.IGNORECASE),
        ],
    ),
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
            re.compile(r"\bmil[-\s]", re.IGNORECASE),
        ],
    ),
    (
        "cruise_book",
        [
            re.compile(r"\bcruise books?\b", re.IGNORECASE),
            re.compile(r"\bcruisebooks?\b", re.IGNORECASE),
            re.compile(r"\bcruise book\b", re.IGNORECASE),
        ],
    ),
    (
        "command_history",
        [
            re.compile(r"\bcommand histories\b", re.IGNORECASE),
            re.compile(r"\bcommand history\b", re.IGNORECASE),
            re.compile(r"\bchr\b", re.IGNORECASE),
        ],
    ),
    (
        "declaration",
        [
            re.compile(r"\bdeclarations?\b", re.IGNORECASE),
        ],
    ),
    (
        "affidavit",
        [
            re.compile(r"\baffidavits?\b", re.IGNORECASE),
            re.compile(r"\baffadavits?\b", re.IGNORECASE),
        ],
    ),
    (
        "deposition",
        [
            re.compile(r"\bdepositions?\b", re.IGNORECASE),
            re.compile(r"\bdepo\b", re.IGNORECASE),
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
            re.compile(r"\bfcr\b", re.IGNORECASE),
            re.compile(r"\bcase[\s_-]*notes?\b", re.IGNORECASE),
        ],
    ),
    (
        "research_document",
        [
            re.compile(r"\bresearch\b", re.IGNORECASE),
        ],
    ),
]

BAD_SHIP_SUFFIXES = {
    "CLASS",
    "DECK",
    "LOG",
    "APRIL",
    "MARCH",
    "JANUARY",
    "FEBRUARY",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
}


def clean_ship_name(ship: str) -> str:
    ship = ship.strip()

    # Remove possessive 's
    ship = re.sub(r"[’']s\b", "", ship)

    # Remove "-class" or "class"
    ship = re.sub(r"-?class\b", "", ship, flags=re.IGNORECASE)

    # Tokenize
    parts = ship.split()

    cleaned = []
    for part in parts:
        if part.upper() in BAD_SHIP_SUFFIXES:
            break
        cleaned.append(part)

    ship = " ".join(cleaned)

    return ship.strip()

def _build_search_space(title: str, text: str = "") -> str:
    return f"{title}\n{text}" if text else title


def _valid_year(y: int) -> bool:
    return MIN_VALID_YEAR <= y <= MAX_VALID_YEAR


def _add_full_year_range(years_found: set[int], start_year: int, end_year: int) -> None:
    for year in range(start_year, end_year + 1):
        years_found.add(year)


def _normalize(s: str) -> str:
    """Normalize string for matching."""
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_ship_class(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.replace("–", "-").replace("—", "-")

    words = []
    for token in raw.split():
        if token.upper() == "USS":
            words.append("USS")
        else:
            parts = token.split("-")
            parts = [p.capitalize() if p else p for p in parts]
            words.append("-".join(parts))

    return " ".join(words) + "-class"


def _normalize_shipyard(raw: str) -> str:
    return " ".join(raw.split()).strip()


def build_rate_patterns(rate_aliases: dict[str, list[str]]):
    compiled = []
    alias_to_canonical = {}

    for canonical, aliases in rate_aliases.items():
        for alias in aliases:
            compiled.append(
                (
                    canonical,
                    alias,
                    re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE),
                )
            )
            alias_to_canonical[_normalize(alias)] = canonical

    return compiled, alias_to_canonical


RATE_PATTERNS, RATE_ALIAS_TO_CANONICAL = build_rate_patterns(RATE_ALIASES)

# Build one combined rate regex so rates can be collected during the shared pass.
# Sort longest-first so more specific aliases win when overlaps exist.
_SORTED_RATE_ALIASES = sorted(
    {alias for aliases in RATE_ALIASES.values() for alias in aliases},
    key=len,
    reverse=True,
)
RATE_MASTER_RE = re.compile(
    r"\b(" + "|".join(re.escape(alias) for alias in _SORTED_RATE_ALIASES) + r")\b",
    re.IGNORECASE,
)


def build_metadata_context(file_path: str | Path, title: str, text: str) -> dict:
    p = Path(file_path)
    text = text or ""
    search_space = _build_search_space(title, text)
    title_len = len(title)

    ctx = {
        "file_path": str(p),
        "path_obj": p,
        "title": title,
        "title_lower": title.lower(),
        "text": text,
        "search_space": search_space,
        "path_parts_normalized": [_normalize(part) for part in p.parts],
        "title_normalized": _normalize(title),
        "title_len": title_len,
    }

    ctx["candidates"] = collect_metadata_candidates(ctx)
    return ctx


def collect_metadata_candidates(ctx: dict) -> dict:
    """
    Shared collection pass over the search space.

    We scan title + text once for raw candidate evidence, then the
    extract_* functions resolve their final answers from this shared structure.
    """
    search_space = ctx["search_space"]
    title_len = ctx["title_len"]

    data = {
        "job_number": None,
        "ships": [],
        "ship_positions": {},
        "ship_classes": [],
        "ship_class_positions": {},
        "years_found": set(),
        "ranges_found": [],
        "shipyards": [],
        "shipyard_positions": {},
        "rates": [],
        "rate_positions": {},
    }

    def region_priority(start: int) -> tuple[int, int]:
        """
        Prefer title hits over text hits, then earlier positions within each region.
        """
        in_title = start < title_len
        return (0 if in_title else 1, start)

    # --- Job number ---
    # Preserve old behavior:
    #   1. title first
    #   2. first JOB_WINDOW chars of text
    m = JOB_RE.search(ctx["title"])
    if m:
        data["job_number"] = m.group(1).upper()
    else:
        m = JOB_RE.search(ctx["text"][:JOB_WINDOW])
        if m:
            data["job_number"] = m.group(1).upper()

    # --- Shared text pass ---
    for match in SHIP_RE.finditer(search_space):
        ship = " ".join(match.group(1).split()).strip()
        ship = clean_ship_name(ship)

        if len(ship.split()) < 2:
            continue
        if ship not in data["ship_positions"]:
            data["ship_positions"][ship] = match.start()

    for match in SHIP_CLASS_RE.finditer(search_space):
        normalized = _normalize_ship_class(match.group(1))
        if normalized not in data["ship_class_positions"]:
            data["ship_class_positions"][normalized] = match.start()

    for match in SHIPYARD_RE.finditer(search_space):
        yard = _normalize_shipyard(match.group(1))
        if yard not in data["shipyard_positions"]:
            data["shipyard_positions"][yard] = match.start()

    for match in RATE_MASTER_RE.finditer(search_space):
        alias = _normalize(match.group(1))
        canonical = RATE_ALIAS_TO_CANONICAL.get(alias)
        if canonical is None:
            continue
        start = match.start()
        if canonical not in data["rate_positions"] or start < data["rate_positions"][canonical]:
            data["rate_positions"][canonical] = start

    for match in YEAR_RANGE_RE.finditer(search_space):
        start_year, end_year = int(match.group(1)), int(match.group(2))
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        if _valid_year(start_year) and _valid_year(end_year):
            data["ranges_found"].append((start_year, end_year, match.start()))
            _add_full_year_range(data["years_found"], start_year, end_year)

    for match in FROM_TO_YEAR_RE.finditer(search_space):
        start_year, end_year = int(match.group(1)), int(match.group(2))
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        if _valid_year(start_year) and _valid_year(end_year):
            data["ranges_found"].append((start_year, end_year, match.start()))
            _add_full_year_range(data["years_found"], start_year, end_year)

    for match in YEAR_RE.finditer(search_space):
        year = int(match.group(1))
        if _valid_year(year):
            data["years_found"].add(year)

    # Finalize ordered lists by best position
    data["ships"] = [
        ship
        for ship, _pos in sorted(
            data["ship_positions"].items(),
            key=lambda item: region_priority(item[1]),
        )
    ]
    data["ship_classes"] = [
        ship_class
        for ship_class, _pos in sorted(
            data["ship_class_positions"].items(),
            key=lambda item: region_priority(item[1]),
        )
    ]
    data["shipyards"] = [
        yard
        for yard, _pos in sorted(
            data["shipyard_positions"].items(),
            key=lambda item: region_priority(item[1]),
        )
    ]
    data["rates"] = [
        canonical
        for canonical, _pos in sorted(
            data["rate_positions"].items(),
            key=lambda item: region_priority(item[1]),
        )
    ]

    data["years_mentioned"] = sorted(data["years_found"])
    data["ranges_found"].sort(key=lambda item: region_priority(item[2]))

    return data


def extract_job_number(ctx: dict) -> str | None:
    return ctx["candidates"]["job_number"]


def classify_doc_type(ctx: dict) -> str:
    candidates = ctx["path_parts_normalized"] + [ctx["title_normalized"]]

    for doc_type, patterns in DOC_TYPE_PATTERNS:
        for candidate in candidates:
            for pattern in patterns:
                if pattern.search(candidate):
                    return doc_type

    return "undefined_doc"


def extract_ships(ctx: dict) -> list[str]:
    return list(ctx["candidates"]["ships"])


def extract_ship_classes(ctx: dict) -> list[str]:
    return list(ctx["candidates"]["ship_classes"])


def extract_year_info(ctx: dict) -> list[int]:
    return list(ctx["candidates"]["years_mentioned"])


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

    if not sample.strip():
        return "ocr_noisy"

    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    words = re.findall(r"\b\w+\b", sample)
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]

    total_words = len(words)
    if total_words == 0:
        return "ocr_noisy"

    weird_char_count = len(re.findall(r"[|~`^_]{1,}", sample))

    junky_words = 0
    for w in alpha_words:
        non_alpha = sum(1 for ch in w if not ch.isalpha())
        if len(w) >= 4 and non_alpha / len(w) > 0.35:
            junky_words += 1

    junk_ratio = junky_words / max(len(alpha_words), 1)
    short_lines = sum(1 for line in lines if len(line) < 20)
    short_line_ratio = short_lines / max(len(lines), 1)

    if weird_char_count > 20 or junk_ratio > 0.18 or short_line_ratio > 0.75:
        return "ocr_noisy"

    return "ocr_clean"


def extract_rates(ctx: dict) -> list[str]:
    return list(ctx["candidates"]["rates"])


def extract_shipyards(ctx: dict) -> list[str]:
    return list(ctx["candidates"]["shipyards"])


def extract_all_metadata(
    file_path: str | Path,
    title: str,
    text: str,
    source: str | None = None,
) -> dict:
    ctx = build_metadata_context(file_path, title, text)

    job_num = extract_job_number(ctx)
    doc_type = classify_doc_type(ctx)
    ships = extract_ships(ctx)
    years_mentioned = extract_year_info(ctx)
    ship_classes = extract_ship_classes(ctx)
    rates_mentioned = extract_rates(ctx)
    shipyards_mentioned = extract_shipyards(ctx)
    source_quality = classify_source_quality(source, ctx["text"]) if source else None

    return {
        "file_path": ctx["file_path"],
        "title": ctx["title"],
        "job_number": job_num,
        "doc_type": doc_type,
        "ships": ships,
        "years_mentioned": years_mentioned,
        "ship_classes": ship_classes,
        "rates_mentioned": rates_mentioned,
        "shipyards_mentioned": shipyards_mentioned,
        "source": source,
        "source_quality": source_quality,
    }