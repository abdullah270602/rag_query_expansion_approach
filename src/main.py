from dotenv import load_dotenv
import os
from chroma_collection import add_to_collection, create_collection, retrieve_documents
from data_processing import read_file, split_into_chunks, split_into_tokens
import groq
from response import expand_query, response_to_original_query
from utils import pretty_print

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

def main():
    # Initialize Groq client
    groq_client = groq.Client(api_key=API_KEY)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "..", "data")

    # Create or retrieve the Chroma collection
    collection_name = "FYP_TESTING"
    collection = create_collection(collection_name)

    # Get a list of all PDF files in the data folder
    files = [
        f for f in os.listdir(test_dir) 
    ]

    # Read, chunk, token, and store each PDF
    for file in files:
        file_path = os.path.join(test_dir, file)
        print(f"Reading : {file}")

        file_text = read_file(file_path)
        

        print(f"Splitting {file} into chunks...")
        chunks = split_into_chunks(file_text)

        print("Splitting chunks into tokens...")
        tokens = split_into_tokens(chunks)
        pretty_print(file_text)
        # Store all tokens in all_tokens
        all_tokens = []
        all_tokens.extend(tokens)

    print(f"Adding documents from files to the collection...")
    add_to_collection(collection, all_tokens)

    print(f"\nAll files have been processed and added to '{collection_name}' collection.")
    print(f"Total documents in collection: {collection.count()}")

    while True:
        # Query Expansion
        user_query = input("\n Enter your query (or type '0' exit): ")
        if user_query.lower() in ["0", "bye"]:
            print("Exiting...")
            break

        expanded_query = expand_query(groq_client, user_query)
        print(f"\nExpanded Query: {expanded_query}")

        print("Retrieving relevant documents...")
        retrieved_docs_expanded = retrieve_documents(collection, expanded_query)
        retrieved_docs_user = retrieve_documents(collection, user_query)
        pretty_print(retrieved_docs_expanded)
        pretty_print(retrieved_docs_user)

        docs_expanded = retrieved_docs_expanded[0] if retrieved_docs_expanded else []
        docs_user = retrieved_docs_user[0] if retrieved_docs_user else []

        context_str = (
            "Documents from the original query:\n"
            + "\n".join(docs_user)
            + "\n\n---\n\n"
            + "Documents from the expanded query:\n"
            + "\n".join(docs_expanded)
        )

        print("Generating a response...")
        response = response_to_original_query(
            groq_client,
            user_query,
            context_str,
        )

        print("\n \nUser Query:", user_query)
        print("-" * 50)
        print("Response: \n", response)
        print("-" * 50)

if __name__ == "__main__":
    main()
