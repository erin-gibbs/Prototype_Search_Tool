import streamlit as st
import fitz  # PyMuPDF
import base64
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- APP CONFIGURATION ---
PDF_MAP = {
    "Amendment 50 & CRS": "PDFs/updatedLGact.pdf",
    "CLGR": "PDFs/1CCR207-1CombinedRules31726.pdf",
    "ICMP": "PDFs/CombinedICMPEffectiveApril12026.pdf",
    "Notifications": "PDFs/NotificationRequirementsDocApril12026"
}
LOGO_PATH = "PDFs/DOGLogo.jpg"

# --- STYLING: RED THEME & CHECKMARKS ---
st.set_page_config(page_title="Colorado Gaming Search", layout="wide")

st.markdown("""
    <style>
    .red-banner { background-color: #C8102E; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }
    .banner-text { color: white; font-weight: bold; font-size: 28px; margin: 0; }
    /* Force Red Checkboxes */
    .stCheckbox > label > div[data-testid="stMarker"] { background-color: #C8102E !important; border-color: #C8102E !important; }
    /* Red Search Button */
    div.stButton > button { background-color: #C8102E; color: white; font-weight: bold; width: 100%; height: 3em; border: none; }
    div.stButton > button:hover { background-color: #A00D25; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE: PDF INDEXING & TF-IDF SEARCH ---
class GamingSearchEngine:
    def __init__(self):
        self.docs = []  # List of {content, source_label, path, page, ref_id}
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def index_files(self):
        """Extracts text from the 4 local PDFs and builds a TF-IDF index."""
        for label, path in PDF_MAP.items():
            try:
                doc = fitz.open(path)
                current_sec = "General"
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    
                    # Chunking text by paragraph for better granularity
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                    for para in paragraphs:
                        # Detection for Reference IDs (e.g., 30-101 or Section 11)
                        ref_match = re.search(r'(\d{2,}-\d{2,}-\d+|SECTION\s+\d+)', para, re.I)
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
                st.error(f"Error indexing {path}: {e}")
        
        if self.docs:
            self.tfidf_matrix = self.vectorizer.fit_transform([d['content'] for d in self.docs])

    def search(self, query, active_sources):
        if self.tfidf_matrix is None: return []
        # Always include Notifications PDF in the search as per requirement
        if "Notifications" not in active_sources:
            active_sources.append("Notifications")
            
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top matches and filter by source
        top_indices = sims.argsort()[-15:][::-1]
        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            if doc['source'] in active_sources and sims[idx] > 0.05:
                results.append(doc)
        return results[:5]

# --- UTILITY: BASE64 PDF OPENER ---
def get_pdf_tab_link(path, page_num):
    """Encodes PDF to Base64 to allow opening a specific local page in a new tab."""
    try:
        with open(path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        # PDF parameters #page=X are supported by most modern browsers
        pdf_link = f'data:application/pdf;base64,{base64_pdf}#page={page_num}'
        return f'<a href="{pdf_link}" target="_blank" style="text-decoration:none;"><button style="background-color:#C8102E; color:white; border:none; padding:8px 16px; border-radius:5px; cursor:pointer; font-weight:bold;">View PDF (Page {page_num})</button></a>'
    except:
        return "File access error."

# --- UI LAYOUT ---
# Header
col_logo, col_banner = st.columns([0.15, 0.85])
with col_logo:
    try: st.image(LOGO_PATH, width=120)
    except: st.write("Logo")
with col_banner:
    st.markdown('<div class="red-banner"><p class="banner-text">Colorado Gaming Regulatory Search</p></div>', unsafe_allow_html=True)

# Search Bar
col_in, col_btn = st.columns([0.85, 0.15])
with col_in:
    query = st.text_input("Query", label_visibility="collapsed", placeholder="Enter rules or keywords (e.g. 'blackjack payout')")
with col_btn:
    search_clicked = st.button("SEARCH")

# Source Selection logic
if 'all_check' not in st.session_state: st.session_state.all_check = True

def toggle_all():
    st.session_state.src_1 = st.session_state.all_val
    st.session_state.src_2 = st.session_state.all_val
    st.session_state.src_3 = st.session_state.all_val

st.write("**Sources:**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    all_cb = st.checkbox("All", value=st.session_state.all_check, key="all_val", on_change=toggle_all)
with c2:
    src_1 = st.checkbox("Amendment 50 & CRS", value=st.session_state.all_check, key="src_1")
with c3:
    src_2 = st.checkbox("CLGR", value=st.session_state.all_check, key="src_2")
with c4:
    src_3 = st.checkbox("ICMP", value=st.session_state.all_check, key="src_3")

# Map UI to sources
active_sources = []
if src_1: active_sources.append("Amendment 50 & CRS")
if src_2: active_sources.append("CLGR")
if src_3: active_sources.append("ICMP")

# --- INITIALIZATION ---
@st.cache_resource
def load_engine():
    engine = GamingSearchEngine()
    engine.index_files()
    return engine

search_engine = load_engine()

# --- RESULTS DISPLAY ---
if (search_clicked or query) and query:
    if not active_sources:
        st.error("Please select at least one source.")
    else:
        results = search_engine.search(query, active_sources)
        
        # Plain Language Summary
        st.markdown("### 📝 Regulatory Summary")
        if results:
            st.info(f"The following regulations were found in your local documents regarding '{query}'. "
                    f"Sources include {', '.join(set([r['source'] for r in results]))}. "
                    "Refer to the direct citations below for specific legal language.")
        else:
            st.warning("No matches found in the embedded PDF files.")

        # Unedited Citations
        st.markdown("### 📜 Official Citations")
        for res in results:
            with st.container():
                st.markdown(f"**{res['source']} | Reference: {res['ref_id']}**")
                # Direct unedited text from PDF
                st.write(res['content'])
                # Base64 View Button
                btn_html = get_pdf_tab_link(res['path'], res['page'])
                st.markdown(btn_html, unsafe_allow_html=True)
                st.markdown("---")
