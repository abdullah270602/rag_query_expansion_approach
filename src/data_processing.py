import os
from pypdf import PdfReader
from docx import Document  # from python-docx
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


def read_file(file_path):
    """
    Function to read different file types based on extension.
    Returns a list of strings, where each string is a chunk of text 
    (e.g., a page, paragraph, or entire file).
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def read_pdf(file_path):
    """ Read a PDF file and return the text content of the file """
    reader = PdfReader(file_path)
    pdf_text = [p.extract_text().strip() for p in reader.pages]
    # Filter out empty strings
    return [text for text in pdf_text if text]

def read_docx(file_path):
    """Read a DOCX file and return the text content as a list of paragraph strings."""
    doc = Document(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    return paragraphs


def read_txt(file_path):
    """Read a TXT file and return the text content as a list with a single element (the entire file)."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return [content.strip()] if content.strip() else []


def split_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """ Split a text into chunks """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text("\n".join(text))


def split_into_tokens(text, tokens_per_chunk=300, chunk_overlap=100):
    """ Split a text into tokens """
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap, tokens_per_chunk=tokens_per_chunk
    )
    # Use the token_splitter to split each chunk into smaller tokens
    tokens = []
    for text in text:
        tokens += splitter.split_text(text)
        
    return tokens