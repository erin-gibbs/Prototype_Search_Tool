import html
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Colorado Gaming Regulatory Search",
    layout="wide",
)

RED = "#B3191F"
LIGHT_RED = "#FCEBEC"
BORDER = "#D7D7D7"
TEXT = "#1F1F1F"

APP_DIR = Path(__file__).resolve().parent
PDF_DIR = APP_DIR / "PDFs"

# Support either Image/ or image/
IMAGE_DIR_CANDIDATES = [APP_DIR / "Image", APP_DIR / "image"]

STATIC_DIR = APP_DIR / "static"
STATIC_PDF_DIR = STATIC_DIR / "PDFs"
STATIC_IMAGE_DIR = STATIC_DIR / "Image"

AMENDMENT_FILE = "updatedLGact.pdf"
NOTIFICATION_FILE = "NotificationRequirementsDocApril12026.pdf"
LOGO_FILE = "DOGLogo.jpg"

# =========================================================
# DATA MODEL
# =========================================================
@dataclass(frozen=True)
class SourceFile:
    source_group: str         # Amendment 50 & CRS / CLGR / ICMP / Notification
    display_label: str        # same as source_group
    path: Path
    filename: str


@dataclass
class PageRecord:
    source_group: str
    display_label: str
    filename: str
    page_number: int
    text: str
    normalized_text: str
    pdf_static_link_base: str
    explicit_citations: List[str] = field(default_factory=list)
    inherited_citations: List[str] = field(default_factory=list)

    @property
    def all_citations(self) -> List[str]:
        seen = set()
        out = []
        for item in self.explicit_citations + self.inherited_citations:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out


# =========================================================
# GENERAL HELPERS
# =========================================================
def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9&/\-]+", text.lower())


def parse_query(query: str) -> Tuple[str, List[str]]:
    quoted_phrases = re.findall(r'"([^"]+)"', query)
    cleaned = re.sub(r'"([^"]+)"', " ", query)
    cleaned = normalize_text(cleaned)
    phrases = [normalize_text(p) for p in quoted_phrases if normalize_text(p)]
    return cleaned, phrases


def find_logo_path() -> Optional[Path]:
    for folder in IMAGE_DIR_CANDIDATES:
        candidate = folder / LOGO_FILE
        if candidate.exists():
            return candidate
    return None


def pdf_link(filename: str, page_number: int) -> str:
    safe_name = quote(filename)
    return f"app/static/PDFs/{safe_name}#page={page_number}"


def ensure_static_assets(source_files: List[SourceFile], logo_path: Optional[Path]) -> None:
    STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for src in source_files:
        dst = STATIC_PDF_DIR / src.filename
        if src.path.exists():
            if not dst.exists() or src.path.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src.path, dst)

    if logo_path and logo_path.exists():
        logo_dst = STATIC_IMAGE_DIR / logo_path.name
        if not logo_dst.exists() or logo_path.stat().st_mtime > logo_dst.stat().st_mtime:
            shutil.copy2(logo_path, logo_dst)


def escape_md(text: str) -> str:
    return text.replace("|", "\\|")


# =========================================================
# SOURCE DISCOVERY
# =========================================================
def discover_source_files() -> Dict[str, List[SourceFile]]:
    if not PDF_DIR.exists():
        return {
            "Amendment 50 & CRS": [],
            "CLGR": [],
            "ICMP": [],
            "Notification": [],
        }

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"), key=lambda p: p.name.lower())

    amendment = []
    clgr = []
    icmp = []
    notification = []

    for pdf in all_pdfs:
        name_lower = pdf.name.lower()

        if pdf.name == AMENDMENT_FILE:
            amendment.append(
                SourceFile(
                    source_group="Amendment 50 & CRS",
                    display_label="Amendment 50 & CRS",
                    path=pdf,
                    filename=pdf.name,
                )
            )
        elif pdf.name == NOTIFICATION_FILE:
            notification.append(
                SourceFile(
                    source_group="Notification",
                    display_label="Notification",
                    path=pdf,
                    filename=pdf.name,
                )
            )
        elif pdf.stem.lower().startswith("rule"):
            clgr.append(
                SourceFile(
                    source_group="CLGR",
                    display_label="CLGR",
                    path=pdf,
                    filename=pdf.name,
                )
            )
        elif pdf.stem.lower().startswith("icmp"):
            icmp.append(
                SourceFile(
                    source_group="ICMP",
                    display_label="ICMP",
                    path=pdf,
                    filename=pdf.name,
                )
            )

    return {
        "Amendment 50 & CRS": amendment,
        "CLGR": clgr,
        "ICMP": icmp,
        "Notification": notification,
    }


# =========================================================
# CITATION EXTRACTION
# =========================================================
RULE_REGEXES = [
    r"\bRule\s+30-\d{3,5}(?:\s*\([^)]{1,20}\))*",
    r"\b30-\d{3,5}(?:\s*\([^)]{1,20}\))*",
]

CRS_REGEXES = [
    r"\bCRS\s+\d{1,3}-\d{1,3}-\d{1,5}(?:\.\d+)?(?:\s*\([^)]{1,20}\))*",
    r"\b\d{1,3}-\d{1,3}-\d{1,5}(?:\.\d+)?(?:\s*\([^)]{1,20}\))*",
]

ICMP_REGEXES = [
    r"\b\d+\([A-Z]\)(?:\(\d+\))?(?:\([a-z]\))?",
    r"\b\d+\([A-Z]\)(?:\([A-Z]\))?(?:\(\d+\))?",
    r"\bSection\s+\d+(?:\s*\([A-Z]\))?(?:\(\d+\))?",
]

AMENDMENT_REGEXES = [
    r"\bAmendment\s+50\b",
]

NOTIFICATION_REGEXES = [
    r"\bNotification\s+Table\b",
]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _clean_rule_citation(match: str) -> str:
    match = normalize_text(match)
    match = re.sub(r"^Rule\s+", "", match, flags=re.I)
    return match


def _clean_icmp_citation(match: str) -> str:
    match = normalize_text(match)
    match = re.sub(r"^Section\s+", "", match, flags=re.I)
    return match


def extract_explicit_citations(source_group: str, text: str) -> List[str]:
    found = []

    if source_group == "CLGR":
        for pattern in RULE_REGEXES:
            for m in re.findall(pattern, text, flags=re.I):
                cleaned = _clean_rule_citation(m)
                if cleaned.startswith("30-"):
                    found.append(cleaned)

    elif source_group == "ICMP":
        for pattern in ICMP_REGEXES:
            for m in re.findall(pattern, text, flags=re.I):
                cleaned = _clean_icmp_citation(m)
                # Keep only likely ICMP section-style values
                if re.match(r"^\d+\([A-Z]\)", cleaned):
                    found.append(cleaned)

    elif source_group == "Amendment 50 & CRS":
        for pattern in AMENDMENT_REGEXES + CRS_REGEXES:
            for m in re.findall(pattern, text, flags=re.I):
                found.append(normalize_text(m))

    elif source_group == "Notification":
        for pattern in NOTIFICATION_REGEXES:
            for m in re.findall(pattern, text, flags=re.I):
                found.append(normalize_text(m))

    return _dedupe_keep_order(found)


def apply_citation_carry_forward(pages: List[PageRecord]) -> List[PageRecord]:
    """
    Carry the last seen citation(s) forward within the same PDF when a page appears
    to continue a section but doesn't repeat the heading / citation number.
    """
    last_seen: List[str] = []
    since_last_explicit = 999

    for page in pages:
        if page.explicit_citations:
            last_seen = page.explicit_citations[:]
            since_last_explicit = 0
            continue

        since_last_explicit += 1

        # Only carry forward for nearby continuation pages.
        if last_seen and since_last_explicit <= 2:
            page.inherited_citations = last_seen[:2]

    return pages


# =========================================================
# PDF EXTRACTION
# =========================================================
@st.cache_data(show_spinner=False)
def extract_pages_from_pdf(pdf_path: str, source_group: str, display_label: str, filename: str) -> List[PageRecord]:
    path = Path(pdf_path)
    records: List[PageRecord] = []

    if not path.exists():
        return records

    with fitz.open(path) as doc:
        temp_pages: List[PageRecord] = []

        for i, page in enumerate(doc, start=1):
            # sort=True gives more natural top-left to bottom-right text order
            text = page.get_text("text", sort=True)
            text = normalize_text(text)

            if not text:
                continue

            explicit = extract_explicit_citations(source_group, text)

            temp_pages.append(
                PageRecord(
                    source_group=source_group,
                    display_label=display_label,
                    filename=filename,
                    page_number=i,
                    text=text,
                    normalized_text=text.lower(),
                    pdf_static_link_base=pdf_link(filename, 1).split("#page=")[0],
                    explicit_citations=explicit,
                )
            )

    return apply_citation_carry_forward(temp_pages)


@st.cache_data(show_spinner=False)
def build_page_index(selected_visible_sources: Tuple[str, ...], discovered_sources: Dict[str, List[SourceFile]]) -> List[PageRecord]:
    page_records: List[PageRecord] = []

    for source_name in selected_visible_sources:
        for src in discovered_sources.get(source_name, []):
            page_records.extend(
                extract_pages_from_pdf(
                    str(src.path),
                    source_group=src.source_group,
                    display_label=src.display_label,
                    filename=src.filename,
                )
            )

    # Notification is always searched
    for src in discovered_sources.get("Notification", []):
        page_records.extend(
            extract_pages_from_pdf(
                str(src.path),
                source_group=src.source_group,
                display_label=src.display_label,
                filename=src.filename,
            )
        )

    return page_records


# =========================================================
# SEARCH / RANKING
# =========================================================
def count_keyword_hits(text: str, terms: List[str]) -> int:
    lowered = text.lower()
    total = 0
    for term in terms:
        if not term.strip():
            continue
        total += len(re.findall(rf"\b{re.escape(term.lower())}\b", lowered))
    return total


def all_terms_present(text: str, terms: List[str]) -> bool:
    if not terms:
        return False
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms if term.strip())


def looks_like_heading_match(text: str, terms: List[str]) -> bool:
    if not terms:
        return False

    chunks = re.split(r"(?<=[.:])\s+|\n", text[:1600])
    early_chunks = [c.strip() for c in chunks[:10] if c.strip()]

    for chunk in early_chunks:
        headingish = (
            len(chunk) <= 160
            and (
                chunk.isupper()
                or bool(re.match(r"^(rule|section|article|part|purpose|definitions|notification)", chunk, re.I))
                or bool(re.match(r"^\d+([.\-)]|\s)", chunk))
            )
        )
        if headingish and any(term.lower() in chunk.lower() for term in terms):
            return True

    return False


def make_raw_snippet(page_text: str, query_terms: List[str], phrases: List[str], window: int = 700) -> str:
    search_items = [p for p in phrases if p] + [t for t in query_terms if t]
    lowered = page_text.lower()

    first_idx = -1
    first_len = 0

    for item in search_items:
        idx = lowered.find(item.lower())
        if idx != -1 and (first_idx == -1 or idx < first_idx):
            first_idx = idx
            first_len = len(item)

    if first_idx == -1:
        snippet = page_text[:window].strip()
        if len(page_text) > window:
            snippet += "..."
        return snippet

    start = max(0, first_idx - window // 2)
    end = min(len(page_text), first_idx + first_len + window // 2)
    snippet = page_text[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(page_text):
        snippet = snippet + "..."

    return snippet


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [normalize_text(p) for p in parts if normalize_text(p)]


def choose_best_description(snippet: str, raw_query: str) -> str:
    query_terms = set(tokenize(raw_query))
    sentences = split_sentences(snippet)

    if not sentences:
        return "Relevant language located in cited source."

    scored = []
    for sent in sentences:
        sent_terms = set(tokenize(sent))
        overlap = len(query_terms.intersection(sent_terms))
        score = overlap

        lowered = sent.lower()
        if any(word in lowered for word in ["must", "shall", "required", "prohibited", "violation", "notification", "report"]):
            score += 2
        if any(word in lowered for word in ["assets", "cash", "chips", "gaming", "licensee", "patron", "surveillance"]):
            score += 1

        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    if len(best) > 240:
        best = best[:237].rstrip() + "..."

    return best


def classify_finding(snippet: str, query: str) -> str:
    text = f"{query} {snippet}".lower()

    if any(x in text for x in ["must", "shall", "required", "requirement", "mandate"]):
        return "General Mandates"
    if any(x in text for x in ["access", "secure", "safeguard", "storage", "escort", "surveillance", "unattended", "lock", "bankroll", "protect"]):
        return "Physical and Operational Safeguards"
    if any(x in text for x in ["notify", "notification", "report", "24 hours"]):
        return "Reporting / Notification"
    if any(x in text for x in ["violation", "penalty", "fine", "revocation", "unsuitable"]):
        return "Consequences of Failure"

    return "Relevant Findings"


def search_pages(records: List[PageRecord], raw_query: str, top_k: int = 20) -> List[dict]:
    cleaned_query, quoted_phrases = parse_query(raw_query)
    query_terms = tokenize(cleaned_query)

    if not records or (not cleaned_query and not quoted_phrases):
        return []

    vector_query = cleaned_query if cleaned_query else " ".join(quoted_phrases)
    corpus = [r.text for r in records]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([vector_query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = []
    for i, record in enumerate(records):
        text = record.text
        lowered = record.normalized_text

        keyword_hits = count_keyword_hits(text, query_terms)
        keyword_score = min(keyword_hits / 10.0, 1.0)

        all_terms_bonus = 1.0 if all_terms_present(text, query_terms) else 0.0

        phrase_hits = sum(lowered.count(p.lower()) for p in quoted_phrases)
        full_query_phrase_hits = lowered.count(cleaned_query.lower()) if cleaned_query else 0
        exact_phrase_bonus = min((phrase_hits * 1.5 + full_query_phrase_hits), 2.0) / 2.0

        heading_bonus = 1.0 if looks_like_heading_match(text, query_terms or quoted_phrases) else 0.0

        citation_bonus = 0.15 if record.all_citations else 0.0

        final_score = (
            tfidf_scores[i] * 0.40
            + keyword_score * 0.15
            + all_terms_bonus * 0.10
            + exact_phrase_bonus * 0.15
            + heading_bonus * 0.10
            + citation_bonus * 0.10
        )

        if final_score <= 0:
            continue

        snippet = make_raw_snippet(text, query_terms, quoted_phrases)

        results.append(
            {
                "record": record,
                "score": float(final_score),
                "tfidf_score": float(tfidf_scores[i]),
                "keyword_hits": keyword_hits,
                "snippet": snippet,
                "category": classify_finding(snippet, raw_query),
                "description": choose_best_description(snippet, raw_query),
                "citations_found": record.all_citations or (
                    ["Notification Table"] if record.source_group == "Notification" else ["Not explicitly identified"]
                ),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def dedupe_results(results: List[dict], max_per_source_group: int = 6) -> List[dict]:
    deduped = []
    seen = set()
    per_group_counts: Dict[str, int] = defaultdict(int)

    for item in results:
        rec = item["record"]
        page_key = (rec.filename, rec.page_number)
        if page_key in seen:
            continue
        if per_group_counts[rec.source_group] >= max_per_source_group:
            continue

        seen.add(page_key)
        per_group_counts[rec.source_group] += 1
        deduped.append(item)

    return deduped


def assign_citation_numbers(results: List[dict]) -> List[dict]:
    numbered = []
    for i, item in enumerate(results, start=1):
        new_item = dict(item)
        new_item["citation_number"] = i
        numbered.append(new_item)
    return numbered


def citation_range_text(nums: List[int]) -> str:
    nums = sorted(set(nums))
    if not nums:
        return ""

    ranges = []
    start = nums[0]
    prev = nums[0]

    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n

    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return "[" + ", ".join(ranges) + "]"


# =========================================================
# SUMMARY GENERATION
# =========================================================
def build_structured_summary(results: List[dict], user_query: str) -> str:
    if not results:
        return "No relevant matches were found in the selected embedded PDF sources."

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in results[:10]:
        grouped[item["category"]].append(item)

    intro_cites = citation_range_text([item["citation_number"] for item in results[:5]])

    lines = []
    lines.append(f"### **Regulatory Summary: {html.escape(user_query.strip())}**")
    lines.append("")
    lines.append(
        f"Based on the selected Colorado gaming sources, the retrieved provisions indicate the following regarding **{html.escape(user_query.strip())}** {intro_cites}."
    )
    lines.append("")

    ordered_categories = [
        "General Mandates",
        "Physical and Operational Safeguards",
        "Reporting / Notification",
        "Consequences of Failure",
        "Relevant Findings",
    ]

    for category in ordered_categories:
        items = grouped.get(category, [])
        if not items:
            continue

        lines.append(f"**{category}:**")
        seen_desc = set()

        for item in items[:3]:
            desc = item["description"].strip()
            key = desc.lower()
            if key in seen_desc:
                continue
            seen_desc.add(key)
            cite = citation_range_text([item["citation_number"]])
            lines.append(f"- {html.escape(desc)} {cite}")

        lines.append("")

    return "\n".join(lines).strip()


# =========================================================
# TABLE / RENDER HELPERS
# =========================================================
def build_associated_cited_content(results: List[dict]) -> pd.DataFrame:
    rows = []

    for item in results:
        rec = item["record"]
        regs = item["citations_found"]
        reg_text = "; ".join(regs[:3]) if regs else "Not explicitly identified"

        rows.append(
            {
                "Regulation / ICMP Number": reg_text,
                "Description": item["description"],
                "Source Document Location": f"{rec.filename} - page {rec.page_number} [Citation {item['citation_number']}]",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["Regulation / ICMP Number", "Description", "Source Document Location"]
        ).reset_index(drop=True)
    return df


# =========================================================
# SESSION STATE / CHECKBOXES
# =========================================================
def initialize_session_state() -> None:
    defaults = {
        "all_sources": True,
        "source_amendment": True,
        "source_clgr": True,
        "source_icmp": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sync_from_all() -> None:
    checked = st.session_state["all_sources"]
    st.session_state["source_amendment"] = checked
    st.session_state["source_clgr"] = checked
    st.session_state["source_icmp"] = checked


def sync_from_individuals() -> None:
    st.session_state["all_sources"] = (
        st.session_state["source_amendment"]
        and st.session_state["source_clgr"]
        and st.session_state["source_icmp"]
    )


def get_selected_visible_sources() -> Tuple[str, ...]:
    selected = []
    if st.session_state["source_amendment"]:
        selected.append("Amendment 50 & CRS")
    if st.session_state["source_clgr"]:
        selected.append("CLGR")
    if st.session_state["source_icmp"]:
        selected.append("ICMP")
    return tuple(selected)


# =========================================================
# STYLING
# =========================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                max-width: 1280px;
                margin: 0 auto;
                color: {TEXT};
            }}

            .banner {{
                background-color: {RED};
                color: white;
                font-weight: 800;
                font-size: 2rem;
                text-align: center;
                padding: 1rem 1.25rem;
                border-radius: 0.55rem;
                margin-top: 0.25rem;
            }}

            .sources-label {{
                font-weight: 700;
                margin-top: 0.35rem;
                margin-bottom: 0.2rem;
            }}

            .summary-box {{
                background: #FFF8F8;
                border: 1px solid {LIGHT_RED};
                border-left: 6px solid {RED};
                border-radius: 0.55rem;
                padding: 1rem;
                margin: 1rem 0 1rem 0;
            }}

            .results-box {{
                background: white;
                border: 1px solid {BORDER};
                border-radius: 0.55rem;
                padding: 1rem;
                margin-bottom: 0.8rem;
            }}

            .citation-label {{
                font-weight: 800;
                color: {RED};
                margin-bottom: 0.35rem;
            }}

            .small-note {{
                color: #666;
                font-size: 0.92rem;
                margin-top: 0.25rem;
                margin-bottom: 0.75rem;
            }}

            div.stButton > button {{
                background-color: {RED};
                color: white;
                font-weight: 700;
                border: none;
                border-radius: 0.45rem;
                min-height: 2.65rem;
                width: 100%;
            }}

            div.stButton > button:hover {{
                background-color: #8E1418;
                color: white;
            }}

            .stCheckbox label {{
                font-weight: 500;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# UI RENDER
# =========================================================
def render_header(logo_path: Optional[Path]) -> None:
    left, right = st.columns([1, 6])

    with left:
        if logo_path and logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

    with right:
        st.markdown(
            '<div class="banner">Colorado Gaming Regulatory Search</div>',
            unsafe_allow_html=True,
        )


def render_search_controls() -> Tuple[bool, str]:
    with st.form("search_form", clear_on_submit=False):
        q_col, b_col = st.columns([5, 1])

        with q_col:
            user_query = st.text_input(
                "Search",
                label_visibility="collapsed",
                placeholder="Enter search text",
            )

        with b_col:
            submitted = st.form_submit_button("Search", use_container_width=True)

    st.markdown('<div class="sources-label">Sources:</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1.5, 1, 1])

    with c1:
        st.checkbox("All", key="all_sources", on_change=sync_from_all)

    with c2:
        st.checkbox("Amendment 50 & CRS", key="source_amendment", on_change=sync_from_individuals)

    with c3:
        st.checkbox("CLGR", key="source_clgr", on_change=sync_from_individuals)

    with c4:
        st.checkbox("ICMP", key="source_icmp", on_change=sync_from_individuals)

    st.markdown(
        '<div class="small-note">Notification Requirements are searched automatically for every query.</div>',
        unsafe_allow_html=True,
    )

    return submitted, user_query


def render_associated_cited_content(df: pd.DataFrame) -> None:
    st.markdown("### **Associated Cited Content**")
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_exact_citation_text(results: List[dict]) -> None:
    st.markdown("### **Exact Citation Text**")

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in results:
        grouped[item["category"]].append(item)

    ordered_categories = [
        "General Mandates",
        "Physical and Operational Safeguards",
        "Reporting / Notification",
        "Consequences of Failure",
        "Relevant Findings",
    ]

    for category in ordered_categories:
        items = grouped.get(category, [])
        if not items:
            continue

        st.markdown(f"#### {category}")

        for item in items:
            rec = item["record"]
            number = item["citation_number"]
            regs = "; ".join(item["citations_found"][:3]) if item["citations_found"] else "Not explicitly identified"
            link = pdf_link(rec.filename, rec.page_number)

            st.markdown(
                f"""
                <div class="results-box">
                    <div class="citation-label">[{number}] {html.escape(rec.display_label)} — {html.escape(rec.filename)} — page {rec.page_number}</div>
                    <div><strong>Regulation / ICMP Number:</strong> {html.escape(regs)}</div>
                    <div><a href="{link}" target="_blank">Open PDF to cited page</a></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Raw extracted citation text only. No paraphrase.
            st.code(item["snippet"], language=None)


def render_missing_files(discovered_sources: Dict[str, List[SourceFile]], logo_path: Optional[Path]) -> None:
    missing_messages = []

    if not discovered_sources["Amendment 50 & CRS"]:
        missing_messages.append(f"PDFs/{AMENDMENT_FILE}")

    if not discovered_sources["CLGR"]:
        missing_messages.append("PDFs/Rules*.pdf")

    if not discovered_sources["ICMP"]:
        missing_messages.append("PDFs/ICMP*.pdf")

    if not discovered_sources["Notification"]:
        missing_messages.append(f"PDFs/{NOTIFICATION_FILE}")

    if not logo_path:
        missing_messages.append("Image/DOGLogo.jpg (or image/DOGLogo.jpg)")

    if missing_messages:
        st.error("Missing required file(s):")
        for msg in missing_messages:
            st.write(f"- {msg}")
        st.stop()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    initialize_session_state()
    inject_css()

    discovered_sources = discover_source_files()
    logo_path = find_logo_path()

    all_source_files = (
        discovered_sources["Amendment 50 & CRS"]
        + discovered_sources["CLGR"]
        + discovered_sources["ICMP"]
        + discovered_sources["Notification"]
    )

    render_missing_files(discovered_sources, logo_path)
    ensure_static_assets(all_source_files, logo_path)
    render_header(logo_path)

    submitted, user_query = render_search_controls()

    if submitted:
        query = user_query.strip()
        if not query:
            st.warning("Please enter search text.")
            st.stop()

        selected_sources = get_selected_visible_sources()
        page_records = build_page_index(selected_sources, discovered_sources)

        if not page_records:
            st.warning("No pages were indexed from the embedded PDF files.")
            st.stop()

        with st.spinner("Searching embedded PDF sources..."):
            raw_results = search_pages(page_records, query, top_k=22)
            results = dedupe_results(raw_results, max_per_source_group=6)
            results = assign_citation_numbers(results)

        if not results:
            st.warning("No relevant matches were found in the selected sources.")
            st.stop()

        summary_md = build_structured_summary(results, query)
        associated_df = build_associated_cited_content(results)

        st.markdown(summary_md)
        st.markdown("---")
        render_associated_cited_content(associated_df)
        st.markdown("---")
        render_exact_citation_text(results)


if __name__ == "__main__":
    main()
