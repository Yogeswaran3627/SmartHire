git status"""
Resume parser module for SmartHire.
Extracts raw text from PDF, DOCX, and TXT files.
Also performs basic structural extraction (email, phone, estimated sections).
"""
import io
import re
from pathlib import Path
from pypdf import PdfReader
from docx import Document

def extract_resume_text(file_obj) -> str:
    """
    Extracts plain text from a file upload (Streamlit UploadedFile) or file path/buffer.
    Supports .pdf, .docx, and .txt.
    """
    # If path passed as string or Path
    if isinstance(file_obj, (str, Path)):
        p = Path(file_obj)
        name = p.name.lower()
        content = p.read_bytes()
    else:
        name = getattr(file_obj, "name", "").lower()
        if hasattr(file_obj, "getvalue"):
            content = file_obj.getvalue()
        elif hasattr(file_obj, "read"):
            content = file_obj.read()
        else:
            raise ValueError("Unsupported file input.")

    if name.endswith(".txt"):
        for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                return content.decode(enc)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
            return "\n\n".join(pages_text).strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

    if name.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text.strip())
            return "\n".join(paragraphs).strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {e}")

    raise ValueError("Unsupported format. Please upload a PDF (.pdf), DOCX (.docx), or Text (.txt) file.")

def extract_contact_info(text: str) -> dict:
    """Extracts email and phone number from resume text using regular expressions."""
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    
    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None
    }
