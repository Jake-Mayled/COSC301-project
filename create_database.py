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
DATA_PATH = "submissions/submission"
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
    # Check if the directory exists and print its contents
    if not os.path.exists(DATA_PATH):
        print(f"Directory does not exist: {DATA_PATH}")
    else:
        print(f"Directory exists. Files in the directory: {os.listdir(DATA_PATH)}")

    #Try's to load directory at DATA_PATH    
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.java", show_progress=True, loader_cls=UnstructuredMarkdownLoader)
        print("DirectoryLoader initialized successfully")
    except Exception as e:
        print(f"Error initializing DirectoryLoader: {e}")
        return None

    #Try's to load all java files in the directory as documents
    try:
        print("Loading documents...")
        docs = loader.load()
        print("Documents loaded successfully")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None
    
    #returns the documents
    return docs

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

