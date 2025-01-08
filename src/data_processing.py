from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


def read_pdf(file_path):
    """ Read a PDF file and return the text content of the file """
    reader = PdfReader(file_path)
    pdf_text = [p.extract_text().strip() for p in reader.pages]
    # Filter out empty strings
    return [text for text in pdf_text if text]


def split_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """ Split a text into chunks """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text("\n".join(text))


def split_into_tokens(text, tokens_per_chunk=256, chunk_overlap=0):
    """ Split a text into tokens """
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap, tokens_per_chunk=tokens_per_chunk
    )
    # Use the token_splitter to split each chunk into smaller tokens
    tokens = []
    for text in text:
        tokens += splitter.split_text(text)
        
    return tokens