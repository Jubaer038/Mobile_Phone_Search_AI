from PyPDF2 import PdfReader
from docx import Document
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(file_bytes: bytes, encoding='utf-8') -> str:
    return file_bytes.decode(encoding, errors='ignore')
