from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from prompts import QUERY_EXPANSION_PROMPT, RESPONSE_PROMPT, USER_ORIGINAL_QUERY
from utils import word_wrap, pretty_print
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import groq
import chromadb
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)



load_dotenv()
API_KEY = os.getenv("API_KEY")
groq_client = groq.Client(api_key=API_KEY)


base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "data", "FYP_Approved_Form.pdf")

print("Reading PDF file...")
reader = PdfReader(file_path)
pdf_text = [p.extract_text().strip() for p in reader.pages]

# Filter out empty strings
pdf_text = [text for text in pdf_text if text]
# print(word_wrap(pdf_text[0]))

print("Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", ", ", " ", ""], 
    chunk_size=1000, 
    chunk_overlap=100
)
split_text = splitter.split_text("\n".join(pdf_text))

# pretty_print("Split Text", word_wrap(split_text[0]))
pretty_print("Total Chunks", len(split_text))

print("Splitting chunks into tokens...")
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

# Use the token_splitter to split each chunk into smaller tokens
splitted_tokens = []
for text in split_text:
    splitted_tokens += token_splitter.split_text(text)


embedding_function = SentenceTransformerEmbeddingFunction()
# pretty_print("Embedding Function", embedding_function(splitted_tokens[10]))

print("Creating Chroma Collection...")
COLLECTION = 'FYP'
chroma = chromadb.Client()
chroma_collection = chroma.create_collection(COLLECTION, embedding_function=embedding_function)

print("Adding documents to the collection...")
# Extract embeddings from the splitted tokens
ids = [str(i) for i in range(len(splitted_tokens))]
chroma_collection.add(ids=ids, documents=splitted_tokens)
count = chroma_collection.count()

pretty_print("Total Documents", count)

# print("Querying the collection...")
# results = chroma_collection.query(query_texts=[QUERY], n_results=3)
# retrived_docs = results["documents"][0]

# # for document in retrived_docs:
# #     print(word_wrap(document))
# #     print("\n")
    
    
    
def expand_query(query):
    
    chat = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": QUERY_EXPANSION_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Original Query: {query}",
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature = 0.7,
    )
    
    return chat.choices[0].message.content

expanded_query = expand_query(USER_ORIGINAL_QUERY)
pretty_print("Expanded Query", expanded_query)

print("Querying the collection with expanded query...")
results = chroma_collection.query(query_texts=[expanded_query], n_results=3)
retrived_docs = results["documents"][0]

# for document in retrived_docs:
#     print(word_wrap(document))
#     print("\n")



def response_to_original_query(user_query, retrieved_docs):
    """
    Generate a response to the original query using the retrieved documents as context.
    """
    context = "\n".join(retrieved_docs)
    
    prompt = RESPONSE_PROMPT.format(context=context, user_query=user_query)

    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses context to answer queries accurately."},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
    )
    
    return response.choices[0].message.content
    
    
response = response_to_original_query(USER_ORIGINAL_QUERY, retrived_docs)
pretty_print("Response", response)