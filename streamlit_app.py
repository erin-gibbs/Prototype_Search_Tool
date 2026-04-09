import streamlit as st
import fitz  # PyMuPDF
import base64
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION & PATHS ---
PDF_MAP = {
    "Amendment 50 & CRS": "PDFs/updatedLGact.pdf",
    "CLGR": "PDFs/1CCR207-1CombinedRules31726.pdf",
    "ICMP": "PDFs/CombinedICMPEffectiveApril12026.pdf",
    "Notification": "PDFs/NotificationRequirementsDocApril12026.pdf"
}
LOGO_PATH = "PDFs/DOGLogo.jpg"

# --- UI STYLING: RED THEME ---
st.set_page_config(page_title="Colorado Gaming Search", layout="wide")

# Custom CSS for Red Banner, Centered Text, and Red Checkmarks
st.markdown("""
    <style>
    .red-banner {
        background-color: #C8102E;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
    }
    .banner-text {
        color: white;
        font-weight: bold;
        font-size: 32px;
        margin: 0;
    }
    /* Force Red Checkbox Marks */
    input[type="checkbox"]:checked { background-color: #C8102E !important; }
    .stCheckbox > label > div[data-testid="stMarker"] {
        background-color: #C8102E !important;
        border-color: #C8102E !important;
    }
    /* Red Search Button with white bold text */
    div.stButton > button {
        background-color: #C8102E;
        color: white;
        font-weight: bold;
        width: 100%;
        height: 3em;
        border: none;
    }
    div.stButton > button:hover { background-color: #A00D25; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE: PDF INDEXING & SEARCH ---
class GamingSearchEngine:
    def __init__(self):
        self.docs = []
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = None

    def index_files(self):
        """Extracts and chunks text from the 4 local PDFs."""
        for label, path in PDF_MAP.items():
            try:
                doc = fitz.open(path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    
                    # Split page into logical paragraphs for better relevance
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                    for para in paragraphs:
                        # Extract Reference IDs (e.g., 30-106 or SECTION 1)
                        ref_match = re.search(r'(\d{2,}-\d{3,}|SECTION\s+\d+)', para, re.I)
                        ref_id = ref_match.group(1) if ref_match else f"Page {page_num + 1}"
                        
                        self.docs.append({
                            "content": para,
                            "source": label,
                            "path": path,
                            "page": page_num + 1,
                            "ref_id": ref_id
                        })
                doc.close()
            except Exception as e:
                st.error(f"Could not index {path}: {e}")
        
        if self.docs:
            self.tfidf_matrix = self.vectorizer.fit_transform([d['content'] for d in self.docs])

    def search(self, query, active_filters):
        """Perform TF-IDF search across indexed documents."""
        if self.tfidf_matrix is None or not query:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Filter indices by chosen sources and a minimum relevance threshold
        top_indices = similarities.argsort()[-20:][::-1]
        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            # Match internal metadata label with the UI checkbox label
            if doc['source'] in active_filters and similarities[idx] > 0.02:
                results.append(doc)
        return results[:5]

# --- UTILITY: BASE64 PDF EMBEDDER ---
def get_pdf_tab_link(path, page_num):
    """Encodes PDF as Base64 to open a local file in a new browser tab at a specific page."""
    try:
        with open(path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        # PDF parameter #page=X is native to browser viewers
        pdf_data = f'data:application/pdf;base64,{base64_pdf}#page={page_num}'
        return f'<a href="{pdf_data}" target="_blank" style="text-decoration:none;"><button style="background-color:#C8102E; color:white; border:none; padding:8px 16px; border-radius:5px; cursor:pointer; font-weight:bold;">View PDF (Page {page_num})</button></a>'
    except:
        return "Source file missing."

# --- INTERFACE LAYOUT ---
# 1. Header Row
col_logo, col_banner = st.columns([0.15, 0.85])
with col_logo:
    try: st.image(LOGO_PATH, width=130)
    except: st.write("Logo Missing")
with col_banner:
    st.markdown('<div class="red-banner"><p class="banner-text">Colorado Gaming Regulatory Search</p></div>', unsafe_allow_html=True)

# 2. Search Field Row
col_in, col_btn = st.columns([0.85, 0.15])
with col_in:
    query = st.text_input("Search terms", label_visibility="collapsed", placeholder="Enter query (e.g. 'blackjack payout' or 'found money')")
with col_btn:
    search_clicked = st.button("SEARCH")

# 3. Source Selection Logic
if 'all_checked' not in st.session_state: st.session_state.all_checked = True

def handle_all_toggle():
    st.session_state.src1 = st.session_state.all_val
    st.session_state.src2 = st.session_state.all_val
    st.session_state.src3 = st.session_state.all_val

st.write("**Sources:**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    all_cb = st.checkbox("All", value=st.session_state.all_checked, key="all_val", on_change=handle_all_toggle)
with c2:
    src1 = st.checkbox("Amendment 50 & CRS", value=st.session_state.all_checked, key="src1")
with c3:
    src2 = st.checkbox("CLGR", value=st.session_state.all_checked, key="src2")
with c4:
    src3 = st.checkbox("ICMP", value=st.session_state.all_checked, key="src3")

# Aggregate selected filters
active_filters = []
if src1: active_filters.append("Amendment 50 & CRS")
if src2: active_filters.append("CLGR")
if src3: active_filters.append("ICMP")
# Notification is always included as an embedded searchable file per instructions
active_filters.append("Notification")

# --- EXECUTION ---
@st.cache_resource
def init_engine():
    engine = GamingSearchEngine()
    engine.index_files()
    return engine

engine = init_engine()

if (search_clicked or query) and query:
    if not any([src1, src2, src3]):
        st.error("Please select a source to search.")
    else:
        with st.spinner("Indexing and searching official documents..."):
            results = engine.search(query, active_filters)
        
        # A. Plain Language Summary
        st.markdown("### 📝 Regulatory Summary")
        if results:
            found_sources = ", ".join(set([res['source'] for res in results]))
            st.info(f"Your search for **'{query}'** yielded results in the following official sources: {found_sources}. "
                    "Below are the specific, unedited citations including the exact legal text and direct links to the documents.")
            
            # B. Citations (Categorized)
            st.markdown("### 📜 Official Citations")
            for res in results:
                with st.expander(f"Source: {res['source']} | Reference ID: {res['ref_id']}", expanded=True):
                    # Unedited Text
                    st.write(res['content'])
                    # Base64 Link Button
                    link_html = get_pdf_tab_link(res['path'], res['page'])
                    st.markdown(link_html, unsafe_allow_html=True)
        else:
            st.warning("No matching rules or regulations found in the embedded files. Please try broader keywords.")
