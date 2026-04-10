from __future__ import annotations

import html
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PATHS
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PDF_DIR = APP_DIR / "PDFs"
IMAGE_DIR = APP_DIR / "image"

STATIC_DIR = APP_DIR / "static"
STATIC_PDF_DIR = STATIC_DIR / "PDFs"
STATIC_IMAGE_DIR = STATIC_DIR / "image"

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Colorado Gaming Regulatory Search",
    layout="wide",
)

RED = "#B3191F"
LIGHT_RED = "#F8DDDF"
TEXT = "#1F1F1F"
BORDER = "#D9D9D9"

VISIBLE_SOURCES: Dict[str, Dict[str, str]] = {
    "Amendment 50 & CRS": {
        "filename": "updatedLGact.pdf",
        "label": "Amendment 50 & CRS",
    },
    "CLGR": {
        "filename": "1CCR207-1CombinedRules31726.pdf",
        "label": "CLGR",
    },
    "ICMP": {
        "filename": "CombinedICMPEffectiveApril12C2026.pdf",
        "label": "ICMP",
    },
}

ALWAYS_INCLUDED_SOURCE = {
    "key": "Notification",
    "filename": "NotificationRequirementsDocApril12026.pdf",
    "label": "Notification",
}

LOGO_FILENAME = "DOGLogo.jpg"

# =========================================================
# DATA MODEL
# =========================================================
@dataclass(frozen=True)
class PageRecord:
    source_key: str
    source_label: str
    filename: str
    page_number: int
    text: str
    normalized_text: str
    static_pdf_path: str


# =========================================================
# UTILITIES
# =========================================================
def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9&\-/]+", text.lower())


def parse_query(query: str) -> Tuple[str, List[str]]:
    quoted_phrases = re.findall(r'"([^"]+)"', query)
    cleaned = re.sub(r'"([^"]+)"', " ", query)
    cleaned = normalize_text(cleaned)
    quoted_phrases = [normalize_text(p) for p in quoted_phrases if normalize_text(p)]
    return cleaned, quoted_phrases


def ensure_static_assets() -> None:
    STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for source in list(VISIBLE_SOURCES.values()) + [ALWAYS_INCLUDED_SOURCE]:
        src = PDF_DIR / source["filename"]
        dst = STATIC_PDF_DIR / source["filename"]
        if src.exists():
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)

    logo_src = IMAGE_DIR / LOGO_FILENAME
    logo_dst = STATIC_IMAGE_DIR / LOGO_FILENAME
    if logo_src.exists():
        if not logo_dst.exists() or logo_src.stat().st_mtime > logo_dst.stat().st_mtime:
            shutil.copy2(logo_src, logo_dst)


def pdf_link(filename: str, page_number: int) -> str:
    quoted_name = quote(filename)
    return f"app/static/PDFs/{quoted_name}#page={page_number}"


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

    segments = re.split(r"(?<=[.:])\s+|\n", text[:1500])
    early_segments = [seg.strip() for seg in segments[:10] if seg.strip()]

    for seg in early_segments:
        headingish = (
            len(seg) <= 160
            and (
                seg.isupper()
                or bool(re.match(r"^(rule|section|article|part|purpose|definitions|internal control|notification)", seg, re.I))
                or bool(re.match(r"^\d+([.\-)]|\s)", seg))
            )
        )
        if headingish and any(term.lower() in seg.lower() for term in terms):
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


def escape_md(text: str) -> str:
    return text.replace("|", "\\|")


# =========================================================
# REGULATION NUMBER EXTRACTION
# =========================================================
RULE_PATTERNS = [
    r"\bRule\s+\d{1,2}-\d{3,5}(?:\s*\([^)]{1,20}\))*",
    r"\b\d{1,2}-\d{3,5}(?:\s*\([^)]{1,20}\))*",
    r"\bICMP\s+Section\s+\d+(?:\s*\([A-Za-z0-9]+\))*",
    r"\bSection\s+\d+(?:\s*\([A-Za-z0-9]+\))*",
    r"\bCRS\s+\d{1,3}-\d{1,3}-\d{1,5}(?:\.\d+)?(?:\s*\([^)]{1,20}\))*",
    r"\bAmendment\s+50\b",
    r"\bNotification\s+Table\b",
]


def extract_regulation_numbers(text: str, source_label: str) -> List[str]:
    found = []

    for pattern in RULE_PATTERNS:
        matches = re.findall(pattern, text, flags=re.I)
        for match in matches:
            cleaned = normalize_text(match)
            if cleaned:
                found.append(cleaned)

    # Special handling for notification doc if no formal number appears
    if source_label == "Notification" and not found:
        found.append("Notification Table")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:5]


# =========================================================
# DESCRIPTION EXTRACTION
# =========================================================
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    return [normalize_text(s) for s in sentences if normalize_text(s)]


def choose_best_description(snippet: str, raw_query: str) -> str:
    query_terms = set(tokenize(raw_query))
    sentences = split_sentences(snippet)

    if not sentences:
        return "Relevant language located in cited source."

    scored = []
    for sentence in sentences:
        sentence_terms = set(tokenize(sentence))
        overlap = len(query_terms.intersection(sentence_terms))
        score = overlap

        # small bonus for obligation language
        lowered = sentence.lower()
        if any(word in lowered for word in ["must", "shall", "required", "prohibited", "may not", "violation", "notification"]):
            score += 2

        scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    if len(best) > 220:
        best = best[:217].rstrip() + "..."

    return best


# =========================================================
# THEMATIC GROUPING
# =========================================================
def classify_finding(snippet: str, query: str) -> str:
    text = f"{query} {snippet}".lower()

    if any(word in text for word in ["must", "shall", "required", "requirement", "mandate"]):
        return "General Requirements"

    if any(word in text for word in ["access", "secure", "safeguard", "storage", "escort", "surveillance", "unattended", "lock", "bankroll", "protection"]):
        return "Operational / Physical Controls"

    if any(word in text for word in ["notify", "notification", "24 hours", "report", "reported"]):
        return "Reporting / Notification"

    if any(word in text for word in ["violation", "penalty", "fine", "revocation", "unsuitable"]):
        return "Consequences / Enforcement"

    return "Relevant Findings"


# =========================================================
# PDF EXTRACTION
# =========================================================
@st.cache_data(show_spinner=False)
def extract_pages_from_pdf(pdf_path: str, source_key: str, source_label: str) -> List[PageRecord]:
    path = Path(pdf_path)
    records: List[PageRecord] = []

    if not path.exists():
        return records

    static_path = pdf_link(path.name, 1).split("#page=")[0]

    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            text = normalize_text(text)

            if not text:
                continue

            records.append(
                PageRecord(
                    source_key=source_key,
                    source_label=source_label,
                    filename=path.name,
                    page_number=i,
                    text=text,
                    normalized_text=text.lower(),
                    static_pdf_path=static_path,
                )
            )

    return records


@st.cache_data(show_spinner=False)
def build_page_index(selected_visible_sources: Tuple[str, ...]) -> List[PageRecord]:
    records: List[PageRecord] = []

    for source_name in selected_visible_sources:
        if source_name in VISIBLE_SOURCES:
            meta = VISIBLE_SOURCES[source_name]
            pdf_path = PDF_DIR / meta["filename"]
            records.extend(
                extract_pages_from_pdf(
                    str(pdf_path),
                    source_key=source_name,
                    source_label=meta["label"],
                )
            )

    always_path = PDF_DIR / ALWAYS_INCLUDED_SOURCE["filename"]
    records.extend(
        extract_pages_from_pdf(
            str(always_path),
            source_key=ALWAYS_INCLUDED_SOURCE["key"],
            source_label=ALWAYS_INCLUDED_SOURCE["label"],
        )
    )

    return records


# =========================================================
# SEARCH / RANKING
# =========================================================
def search_pages(records: List[PageRecord], raw_query: str, top_k: int = 18) -> List[dict]:
    cleaned_query, quoted_phrases = parse_query(raw_query)
    query_terms = tokenize(cleaned_query)

    if not records:
        return []

    if not cleaned_query and not quoted_phrases:
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

        quoted_phrase_hits = sum(lowered.count(p.lower()) for p in quoted_phrases)
        full_query_phrase_hits = lowered.count(cleaned_query.lower()) if cleaned_query else 0
        exact_phrase_bonus = min((quoted_phrase_hits * 1.5 + full_query_phrase_hits), 2.0) / 2.0

        heading_bonus = 1.0 if looks_like_heading_match(text, query_terms or quoted_phrases) else 0.0

        final_score = (
            tfidf_scores[i] * 0.45
            + keyword_score * 0.15
            + all_terms_bonus * 0.10
            + exact_phrase_bonus * 0.20
            + heading_bonus * 0.10
        )

        if final_score <= 0:
            continue

        snippet = make_raw_snippet(text, query_terms, quoted_phrases)
        regulations = extract_regulation_numbers(text, record.source_label)

        results.append(
            {
                "record": record,
                "score": float(final_score),
                "tfidf_score": float(tfidf_scores[i]),
                "keyword_hits": keyword_hits,
                "snippet": snippet,
                "all_terms_bonus": all_terms_bonus,
                "exact_phrase_bonus": exact_phrase_bonus,
                "heading_bonus": heading_bonus,
                "regulations": regulations,
                "description": choose_best_description(snippet, raw_query),
                "category": classify_finding(snippet, raw_query),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def dedupe_results(results: List[dict], max_per_source: int = 6) -> List[dict]:
    deduped = []
    seen_pages = set()
    per_source_counts: Dict[str, int] = {}

    for item in results:
        record = item["record"]
        page_key = (record.filename, record.page_number)

        if page_key in seen_pages:
            continue

        if per_source_counts.get(record.source_label, 0) >= max_per_source:
            continue

        seen_pages.add(page_key)
        per_source_counts[record.source_label] = per_source_counts.get(record.source_label, 0) + 1
        deduped.append(item)

    return deduped


# =========================================================
# CITATION NUMBERING
# =========================================================
def assign_citation_numbers(results: List[dict]) -> List[dict]:
    numbered = []
    for i, item in enumerate(results, start=1):
        new_item = dict(item)
        new_item["citation_number"] = i
        numbered.append(new_item)
    return numbered


def citation_range_text(nums: List[int]) -> str:
    if not nums:
        return ""
    nums = sorted(set(nums))
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
    for item in results[:8]:
        grouped[item["category"]].append(item)

    intro_sources = []
    for item in results[:5]:
        intro_sources.append(item["citation_number"])
    intro_cites = citation_range_text(intro_sources)

    lines = []
    lines.append(f"### **Regulatory Summary: {user_query.strip()}**")
    lines.append("")
    lines.append(
        f"Based on the selected Colorado gaming regulatory sources, the retrieved provisions indicate the following regarding **{user_query.strip()}** {intro_cites}."
    )
    lines.append("")

    preferred_order = [
        "General Requirements",
        "Operational / Physical Controls",
        "Reporting / Notification",
        "Consequences / Enforcement",
        "Relevant Findings",
    ]

    for category in preferred_order:
        items = grouped.get(category, [])
        if not items:
            continue

        lines.append(f"**{category}:**")
        seen_desc = set()

        for item in items[:3]:
            desc = item["description"].strip()
            if desc.lower() in seen_desc:
                continue
            seen_desc.add(desc.lower())
            cite = citation_range_text([item["citation_number"]])
            lines.append(f"- {desc} {cite}")

        lines.append("")

    return "\n".join(lines).strip()


# =========================================================
# ASSOCIATED CITED CONTENT TABLE
# =========================================================
def build_associated_cited_content(results: List[dict]) -> pd.DataFrame:
    rows = []

    for item in results:
        rec = item["record"]
        regs = item["regulations"] if item["regulations"] else ["Not explicitly identified"]
        reg_text = "; ".join(regs[:3])

        rows.append(
            {
                "Regulation / ICMP Number": reg_text,
                "Description": item["description"],
                "Source Document Location": f"{rec.filename} - page {rec.page_number} [Citation {item['citation_number']}]",
            }
        )

    df = pd.DataFrame(rows)

    # Deduplicate visually similar rows
    if not df.empty:
        df = df.drop_duplicates(
            subset=["Regulation / ICMP Number", "Description", "Source Document Location"]
        ).reset_index(drop=True)

    return df


# =========================================================
# SESSION STATE / CHECKBOX LOGIC
# =========================================================
def initialize_session_state() -> None:
    defaults = {
        "all_sources": True,
        "source_amendment": True,
        "source_clgr": True,
        "source_icmp": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_from_all() -> None:
    checked = st.session_state["all_sources"]
    st.session_state["source_amendment"] = checked
    st.session_state["source_clgr"] = checked
    st.session_state["source_icmp"] = checked


def sync_from_individuals() -> None:
    all_checked = (
        st.session_state["source_amendment"]
        and st.session_state["source_clgr"]
        and st.session_state["source_icmp"]
    )
    st.session_state["all_sources"] = all_checked


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
                border-radius: 0.5rem;
                margin-top: 0.25rem;
            }}

            .sources-label {{
                font-weight: 700;
                margin-top: 0.25rem;
            }}

            .summary-box {{
                background: #FFF8F8;
                border: 1px solid {LIGHT_RED};
                border-left: 6px solid {RED};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-top: 1rem;
                margin-bottom: 1rem;
            }}

            .results-box {{
                background: white;
                border: 1px solid {BORDER};
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 0.75rem;
            }}

            .citation-label {{
                font-weight: 800;
                color: {RED};
                margin-bottom: 0.35rem;
            }}

            .small-note {{
                color: #666;
                font-size: 0.92rem;
            }}

            div.stButton > button {{
                background-color: {RED};
                color: white;
                font-weight: 700;
                border: none;
                border-radius: 0.45rem;
                min-height: 2.6rem;
                width: 100%;
            }}

            div.stButton > button:hover {{
                background-color: #8F1519;
                color: white;
            }}

            .stCheckbox label {{
                font-weight: 500;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}

            th, td {{
                border: 1px solid #DDD;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }}

            th {{
                background-color: #F7F7F7;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# RENDER
# =========================================================
def render_header() -> None:
    logo_col, banner_col = st.columns([1, 6])

    with logo_col:
        logo_path = IMAGE_DIR / LOGO_FILENAME
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

    with banner_col:
        st.markdown(
            '<div class="banner">Colorado Gaming Regulatory Search</div>',
            unsafe_allow_html=True,
        )


def render_search_controls() -> Tuple[bool, str]:
    with st.form("search_form", clear_on_submit=False):
        query_col, button_col = st.columns([5, 1])

        with query_col:
            user_query = st.text_input(
                "Search",
                label_visibility="collapsed",
                placeholder="Enter search text",
            )

        with button_col:
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
        '<div class="small-note">Notification requirements are searched automatically for every query.</div>',
        unsafe_allow_html=True,
    )

    return submitted, user_query


def render_structured_summary(summary_markdown: str) -> None:
    st.markdown(
        f'<div class="summary-box">{summary_markdown}</div>',
        unsafe_allow_html=True,
    )


def render_associated_cited_content(df: pd.DataFrame) -> None:
    st.markdown("### **Associated Cited Content**")
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_exact_citation_text(results: List[dict]) -> None:
    st.markdown("### **Exact Citation Text**")

    for item in results:
        rec = item["record"]
        number = item["citation_number"]
        link = pdf_link(rec.filename, rec.page_number)
        regs = "; ".join(item["regulations"][:3]) if item["regulations"] else "Not explicitly identified"

        st.markdown(
            f"""
            <div class="results-box">
                <div class="citation-label">[{number}] {html.escape(rec.source_label)} — Page {rec.page_number}</div>
                <div><strong>Regulation / ICMP Number:</strong> {html.escape(regs)}</div>
                <div><a href="{link}" target="_blank">Open PDF to cited page</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Raw citation text only
        st.code(item["snippet"], language=None)


def render_no_results() -> None:
    st.warning("No relevant matches were found in the selected sources.")


def render_missing_files(missing_files: List[str]) -> None:
    st.error("Missing required file(s):")
    for f in missing_files:
        st.write(f"- {f}")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    initialize_session_state()
    ensure_static_assets()
    inject_css()
    render_header()

    required_files = [
        PDF_DIR / "updatedLGact.pdf",
        PDF_DIR / "1CCR207-1CombinedRules31726.pdf",
        PDF_DIR / "CombinedICMPEffectiveApril12C2026.pdf",
        PDF_DIR / "NotificationRequirementsDocApril12026.pdf",
        IMAGE_DIR / "DOGLogo.jpg",
    ]

    missing = [str(p.relative_to(APP_DIR)) for p in required_files if not p.exists()]
    if missing:
        render_missing_files(missing)
        st.stop()

    submitted, user_query = render_search_controls()

    if submitted:
        query = user_query.strip()
        if not query:
            st.warning("Please enter search text.")
            st.stop()

        selected_sources = get_selected_visible_sources()
        page_records = build_page_index(selected_sources)

        if not page_records:
            st.warning("No pages were indexed from the embedded PDF files.")
            st.stop()

        with st.spinner("Searching embedded PDF sources..."):
            raw_results = search_pages(page_records, query, top_k=20)
            results = dedupe_results(raw_results, max_per_source=6)
            results = assign_citation_numbers(results)

        if not results:
            render_no_results()
            st.stop()

        summary_markdown = build_structured_summary(results, query)
        associated_df = build_associated_cited_content(results)

        st.markdown(summary_markdown)
        st.markdown("---")
        render_associated_cited_content(associated_df)
        st.markdown("---")
        render_exact_citation_text(results)


if __name__ == "__main__":
    main()
