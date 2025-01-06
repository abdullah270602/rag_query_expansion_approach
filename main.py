import pprint

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from utils import word_wrap, pretty_print
from pypdf import PdfReader
import groq
from dotenv import load_dotenv
import os

import chromadb
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)


load_dotenv()
API_KEY = os.getenv("API_KEY")


reader = PdfReader("data\microsoft-annual-report.pdf")
pdf_text = [p.extract_text().strip() for p in reader.pages]


# Filter out empty strings
pdf_text = [text for text in pdf_text if text]

# print(word_wrap(pdf_text[0]))

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", ", ", " ", ""], 
    chunk_size=1000, 
    chunk_overlap=100
)
split_text = splitter.split_text("\n".join(pdf_text))

pretty_print("Split Text", word_wrap(split_text[0]))
pretty_print("Total Chunks", len(split_text))


token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

# Use the token_splitter to split each chunk into smaller tokens
splitted_tokens = []
for text in split_text:
    splitted_tokens += token_splitter.split_text(text)


embedding_function = SentenceTransformerEmbeddingFunction()
# pretty_print("Embedding Function", embedding_function(splitted_tokens[10]))


chroma = chromadb.Client()
chroma_collection = chroma.create_collection("report", embedding_function=embedding_function)


# Extract embeddings from the splitted tokens
ids = [str(i) for i in range(len(splitted_tokens))]
chroma_collection.add(ids=ids, documents=splitted_tokens)
count = chroma_collection.count()

pretty_print("Total Documents", count)

QUERY = "what was the total revenue for the year?"

results = chroma_collection.query(query_texts=[QUERY], n_results=3)
retrived_docs = results["documents"][0]

for document in retrived_docs:
    print(word_wrap(document))
    print("\n")