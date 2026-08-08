import io
from pypdf import PdfReader
from docx import Document
import pandas as pd


async def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Extracts plain text content from uploaded file based on file extension.
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext == "pdf":
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        extracted_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)
        return "\n\n".join(extracted_text)

    elif ext in ["docx", "doc"]:
        doc = Document(io.BytesIO(file_bytes))
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    elif ext in ["csv", "xlsx", "xls"]:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        return df.to_string(index=False)

    elif ext in ["txt", "md", "json"]:
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file format: .{ext}")