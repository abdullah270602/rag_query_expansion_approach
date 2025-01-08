from dotenv import load_dotenv
import os
from chroma_collection import add_to_collection, create_collection, retrieve_documents
from data_processing import read_pdf, split_into_chunks, split_into_tokens
import groq
from response import expand_query, response_to_original_query

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

def main():
    # Initialize Groq client
    groq_client = groq.Client(api_key=API_KEY)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "FYP_Approved_Form.pdf")
    
    collection_name = "Test"
    
    print("Reading PDF...")
    pdf_text = read_pdf(file_path)
    
    print("Splitting text into chunks...")
    chunks = split_into_chunks(pdf_text)
    print("Total Chunks", len(chunks))
    
    print("Splitting chunks into tokens...")
    tokens = split_into_tokens(chunks)
    
    print("Creating Chroma Collection...")
    collection = create_collection(collection_name)
    
    print("Adding documents to the collection...")
    collection = add_to_collection(collection, tokens)
    print(f"Total Documents in Collection: {collection.count()}")
    
    while True:
        # Query Expansion
        user_query = input("\n Enter your query (or type '0' exit): ")
        if user_query.lower() in ["0", "bye"]:
            print("Exiting...")
            break
        
        expanded_query = expand_query(groq_client, user_query)
        print(f"\nExpanded Query: {expanded_query}")
        
        print("Retrieving relevant documents...")
        retrieved_docs = retrieve_documents(collection, expanded_query)

        print("Generating a response...")
        response = response_to_original_query(
            groq_client,
            user_query,
            retrieved_docs,
        )
        
        print("\n \nUser Query:", user_query)
        print("-" * 50)
        print("Response: \n", response)
        print("-" * 50)
    
if __name__ == "__main__":
    main()