# helper_utils.py
import numpy as np
import chromadb
import pandas as pd
from pypdf import PdfReader
import numpy as np



def project_embedings(embeddings, umap_transform):
    """
    Project embeddings to 2D space using PCA.
    """
    project_embedings  = umap_transform.transform(embeddings)
    return project_embedings


def word_wrap(text, width=80):
    """
    Wraps the given text to the specified width.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from the given PDF file.
    """
    text = []
    with open(pdf_path, "rb") as file:
        pdf = PdfReader(file)

        for page_num in range(pdf.getNumPages()):
            page = pdf.getPage(page_num)
            text.append(page.extract_text())
        
    return "\n".join(text)


def load_chroma(filename, collection_name, embedding_function):
    """
    Load chroma features from the given file and store them in the database.
    """
    # Extract text from the PDF
    text = extract_text_from_pdf(filename)
    
    # Split text into paragraphs or chunks
    paragraphs = text.split("\n\n")

    # Generate embeddings for each chunk
    embeddings = [embedding_function(paragraph) for paragraph in paragraphs]
    
    # Create a DataFrame to store text and embeddings
    data = {"text": paragraphs, "embeddings": embeddings}
    df = pd.DataFrame(data)

    collection = chromadb.get_or_create_collection(collection_name)
    
    for ids, row in df.iterrows():
        collection.add(ids=ids, document=row["text"], embeddings=row["embeddings"])
        
        
    return collection



def pretty_print(message="", text=""):
    """
    Pretty prints the given text with an optional message.
    """
    print("-" * 50)
    if message:
        print(f"{message}:")
    print(text)
    print("-" * 50)