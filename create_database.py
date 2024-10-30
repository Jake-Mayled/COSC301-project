#Import statements
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
import os
from langchain_community.vectorstores import Chroma
import shutil
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

#Data file path
DATA_PATH = "submissions"
#Chroma DB path
CHROMA_PATH = "chroma"
#Loads .env file
load_dotenv()
#Sets API key
api_key = os.getenv("OPENAI_API_KEY")


def main():
    # Gets documents from directory
    documents = loadDocuments()
    #Saves documents to chroma DB
    save_to_chroma(documents)


#Loads java files as documents from the file path
def loadDocuments():
    all_docs = []

    # Check if the directory exists and print its contents
    if not os.path.exists(DATA_PATH):
        print(f"Directory does not exist: {DATA_PATH}")
        return all_docs
    else:
        print(f"Directory exists. Subdirectories in the directory: {os.listdir(DATA_PATH)}")

    # Iterate over each subdirectory in DATA_PATH
    for submission_dir in os.listdir(DATA_PATH):
        submission_path = os.path.join(DATA_PATH, submission_dir)
        
        if os.path.isdir(submission_path):
            try:
                # Initialize loader for the current submission directory
                loader = DirectoryLoader(submission_path, glob="*.java", show_progress=True, loader_cls=UnstructuredMarkdownLoader)
                print(f"Loading documents from {submission_dir}...")
                
                # Load documents and add directory name as metadata
                docs = loader.load()
                for doc in docs:
                    doc.metadata['submission_directory'] = submission_dir  # Adding directory name as metadata
                
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} documents from {submission_dir}")
            
            except Exception as e:
                print(f"Error loading documents from {submission_dir}: {e}")

    return all_docs

#Saves documents to Chroma DB
def save_to_chroma(docs: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    Chroma.from_documents(
        docs, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(docs)} documents to {CHROMA_PATH}.")

#File construct
if __name__ == "__main__":
    main()

