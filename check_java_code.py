import os
import subprocess
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

CHROMA_PATH = "chroma"
#Loads .env file
load_dotenv()
#Sets API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
embedding_function = OpenAIEmbeddings()

#Marking rubric for java files
with open("marking_rubric.txt") as file:
    rubric = file.read()

def main():
    javaFiles = getJavaFiles()
    response = evaluateCodeWithRubric(javaFiles)
    print(response)

#Gets all java files from chroma DB
def getJavaFiles():
    # Initialize the Chroma vector store
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #Gets java files from the database
    retrieved_documents = vector_store.similarity_search("java files", k=10)
    if not retrieved_documents:
        print("No Java files found in database")
        exit(1)
    else:
        print("Java files retrieved from database")
        return retrieved_documents

#Compiles all java files and checks for any errors
def compileJavaFiles(javaFiles : list[Document]):
    # Extract file paths from the document metadata
    file_paths = [file.metadata['source'] for file in javaFiles]

    # Compile all the Java files together using a single javac command
    print(f"Compiling the following Java files: {file_paths}")
    compilation_process = subprocess.run(["javac"] + file_paths, capture_output=True, text=True)

    # Check for compilation errors
    if compilation_process.returncode != 0:
        print(f"Compilation failed with the following error:\n{compilation_process.stderr}")
        return 1
    else:
        print("Compilation successful for all provided Java files.")
        return 0

#Compiles code, if unsuccessful returns message, else sends java code to OpenAI gpt model and returns the response message
def evaluateCodeWithRubric(javaFiles):
    # compiles java code
    compilation_result = compileJavaFiles(javaFiles)
    #If java code does not compile return with a unsuccessful message
    if compilation_result == 1:
        return "Java code complied unsuccessfully, mark: 0/10"
    
    #Joins all code files together
    code_text = "\n\n".join([file.page_content for file in javaFiles])
    
    #Sends the rubric and the java code to the gpt model and saves the response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Evaluate the following Java code according to this marking rubric:\n\n{rubric}\n\nJava Code:\n{code_text}\n\nReturn a mark and feedback.",
        }]
    )
    
    #Returns the response message content
    return response.choices[0].message.content


#File construct
if __name__ == "__main__":
    main()
