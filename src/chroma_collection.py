import chromadb
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)

def create_collection(collection_name):
    """ Create a collection """
    chroma = chromadb.Client()
    embedding_function = SentenceTransformerEmbeddingFunction()
    collection = chroma.create_collection(collection_name, embedding_function=embedding_function)
    
    return collection


def add_to_collection(collection,tokens):
    """ Add documents to the collection """
    # Extract embeddings from the splitted tokens
    ids = [str(i) for i in range(len(tokens))]
    collection.add(ids=ids, documents=tokens)
    
    return collection


def retrieve_documents(collection,query,n_results=3):
    """ Query the collection with the query """
    results = collection.query(query_texts=[query], n_results=n_results)
    
    return results["documents"]