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
RUBRIC = """
Evaluate the Java code according to the following marking rubric:

1. [1 mark] I have chosen appropriate datatypes and have created enums where appropriate. Any enums contain appropriate switch statements to provide a visually pleasing representation of the Enum constants.
2. [2 marks] I have created a class named Coffee, which contains:
    1) all the features that represent a coffee and/or are used to search the menu including milk, number of shots, sugar and extras. 
    2) appropriate constructor/s, setters and getters.
    3) a method that describes the menu item in the format required by the Greek Geek.
My Coffee class is used to create BOTH 'real' coffees from the menu AND 'dream' coffees (representing the user's filters).
3. [0.5 marks] I have created a record named Geek to represent the app user (name and phone number).
4. [2 marks] I have created a class named Menu which contains:
    1) an appropriate field (data structure) to store and access Coffee objects.
    2) a method to add Coffee objects to the data structure
    3) a method to compare coffees in the field (data structure) to a user's 'dream' coffee (parameter), returning an appropriate collection of matching Coffees.
5. I have created a class named MenuSearcher that contains:
    1) [1 mark] a method to load the data from menu.txt, returning an instance of the Menu class. 
    2) [1 mark] a method that requests user input/selection of coffee features e.g. milk type (including a no-milk option), number of shots, sugar (yes/no), price range and extras), returning a Coffee object representing the user's 'dream' coffee.
    3) [0.5 marks] a method that uses the user's 'dream' Coffee object to search the menu, presenting matches to the user, and allowing selection of a menu item.
    4) [0.5 marks] a method to obtain the user's information (name and phone number) returning a Geek object, as well as relevant helper methods.
    5) [0.5 marks] a method to write the Geek user's order information to a text file in the format specified by the Greek Geek.
    6) [0.5 marks] a main method used to run the program by calling the above methods as appropriate.
    7) [0.5 marks] code to handle when 1) matches are found 2) no matches are found AND 3) the user closes the dialog.

To achieve full marks in the above sections, please ensure you address each of the following:
    I have ensured that all mutable types are protected. [Marks deductible if not addressed: 1]
    My program validates all user input and I've handled all exceptions. [Marks deductible if not addressed: 2]
    My code is thoroughly documented, with Javadoc blocks for each method. [Marks deductible if not addressed: 2]
    My program produces the correct output when the test prompts are entered (see table and video below). [Marks deductible if not addressed: 2]


Provide a mark /10 and feedback based on the above points and indicate any areas for deduction
"""

def main():
    javaFiles = getJavaFiles()
    #compileJavaFiles(javaFiles)
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
            "content": f"Evaluate the following Java code according to this marking rubric:\n\n{RUBRIC}\n\nJava Code:\n{code_text}\n\nReturn a mark and feedback.",
        }]
    )
    
    #Returns the response message content
    return response.choices[0].message.content


#File construct
if __name__ == "__main__":
    main()
