import streamlit as st
from backend import text_to_chunks, create_or_load_chroma, create_llm, create_conversational_chain
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="Mobile Search AI", layout="wide")
st.title("📱 Mobile Search AI with RAG + LangChain")

# Theme switcher
theme = st.radio("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
        """,
        unsafe_allow_html=True
    )

uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
user_question = st.text_input("Ask a question:")

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'chain' not in st.session_state:
    st.session_state.chain = None

def load_text(file):
    """Load text from PDF, TXT, DOCX"""
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    else:
        return ""

# Process uploaded file
if uploaded_file:
    raw_text = load_text(uploaded_file)
    chunks = text_to_chunks(raw_text)
    st.session_state.vectordb = create_or_load_chroma(chunks)
    st.session_state.chain, st.session_state.memory = create_conversational_chain(st.session_state.vectordb)
    st.success("✅ File loaded and vector DB created!")

# Process user question
if user_question and st.session_state.chain:
    with st.spinner("Generating answer..."):
        result = st.session_state.chain.run(user_question)
    st.markdown("**Answer:**")
    st.write(result)
