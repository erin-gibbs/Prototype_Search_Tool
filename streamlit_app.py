import streamlit as st
import pymupdf as fitz
import os

# 1. Branding Header
st.set_page_config(page_title="Colorado Gaming Search", layout="wide")
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("PDFs/DOGLogo.jpg"):
        st.image("PDFs/DOGLogo.jpg", width=120)
with col2:
    st.markdown("<h1 style='color: #CC0000;'>Colorado Gaming Regulatory Search</h1>", unsafe_allow_html=True)

# 2. Sidebar/Sources
st.sidebar.header("Search Sources")
all_sources = st.sidebar.checkbox("All Sources", value=True)

sources = {
    "Amendment 50 & CRS": "PDFs/updatedLGact.pdf",
    "CLGR": "PDFs/1CCR207-1CombinedRules31726.pdf",
    "ICMP": "PDFs/CombinedICMPEffectiveApril1%2C2026_0.pdf"
}

selected_paths = []
for name, path in sources.items():
    if all_sources or st.sidebar.checkbox(name, value=False):
        selected_paths.append(path)

# 3. Search Interface
query = st.text_input("Enter search term (e.g., 'Self Exclusion')", placeholder="Type here...")
search_button = st.button("Search")
if search_button and query:
    if query:
        st.write(f"Searching for: **{query}**...")
        
        for path in selected_paths:
            if os.path.exists(path):
                doc = fitz.open(path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if query.lower() in text.lower():
                        with st.expander(f"Match found in {path.split('/')[-1]} (Page {page_num + 1})"):
                            st.info(f"**Source:** {path}")
                            st.write(text[:500] + "...") # Shows a snippet
            else:
                st.error(f"File not found: {path}")
