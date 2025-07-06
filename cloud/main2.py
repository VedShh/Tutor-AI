#Imports for Baseline QA Pipeline
import subprocess
from langchain_community.document_loaders import PyPDFLoader # for loading the pdf
from langchain_openai import OpenAIEmbeddings # for creating embeddings
from langchain_community.vectorstores import Chroma # for the vectorization part
from langchain.chains import RetrievalQA #For the retrieval QA chain part # apparently deprecated
from langchain_openai import ChatOpenAI #for getting an LLM for QA chain
#from langchain_core.output_parsers import StrOutputParser #Not used currently, leaving, as can be used for parsing output from LLM
#from langchain_core.runnables import RunnablePassthrough #Not used currently, leaving, as can be used for getting LLM output
from langchain.prompts import ChatPromptTemplate #for setting up prompts
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import OpenAI

#Setup openai key
import os
from dotenv import load_dotenv
load_dotenv() 

class HistoryChatBot:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-4o', temperature=0.6)

        #Setup a prompt template
        template = """\
        You are an assistant for question-answering tasks.

        Use the following pieces of retrieved context to answer the question.

        If either the PDF or the question are not related to each other or not 
        related to any educational standard, state the following: This content is 
        not related to any educational purposes. 

        For example, if topics are not the same, like a java textbook is given, 
        however, the user asks about a physics question, state the following: This
        content is not related to the inputted textbook, please select another textbook
        and try again.

        If you don't know the answer, just say that you don't know.

        Use three sentences maximum and keep the answer concise. 

        Question: {question}

        Context: {context}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}



        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def invoke(self, input_dict):
        question = input_dict.get("question")
        print(question)
        context += ("Question: " + question)

        result = self.qa_chain.invoke({
            "query": question,
            "context": context
        })

        self.update_chat_history(question, result['result'])
        
        context += ("Result: " + result)
        return result

#Setup Base QA system pipeline
class BaseQAPipeline:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-4o', temperature=0.6)

        #Setup a prompt template
        template = """\
        You are an assistant for question-answering tasks.

        Use the following pieces of retrieved context to answer the question.

        If either the PDF or the question are not related to each other or not 
        related to any educational standard, state the following: This content is 
        not related to any educational purposes. 

        For example, if topics are not the same, like a java textbook is given, 
        however, the user asks about a physics question, state the following: This
        content is not related to the inputted textbook, please select another textbook
        and try again.

        If you don't know the answer, just say that you don't know.

        Use three sentences maximum and keep the answer concise. 

        Question: {question}

        Context: {context}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}



        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        print(question)

        combined_context = self.build_combined_context()

        context_from_db = self.vectorstore_retriever.get_relevant_documents(question)
        
        # Extract page numbers
        page_numbers = [doc.metadata.get("page", "Unknown") for doc in context_from_db]
        
        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])

        # Include page number in the result
        return {'query': question, 'context': combined_context, 'result': result['result'], 'pages': page_numbers}


      
# #Setup GenerateStudyPlan pipeline
class GenerateStudyPlan:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-4o', temperature=0.6)

        #Setup a prompt template
        template = """\
        You are an expert in making highly customized and personilized study plans. A user has asked you to make
        then a study plan, and you will give them your best work. Feel free to use the user's data to make their
        study plan more personilized. Make sure to revolve the study plan around the user's weaknesses. Create an
        outline for this plan and elaborate on certain parts where you think that the user needs more help. 

        If either the PDF or the question are not related to each other or not 
        related to any educational standard, state the following: This content is 
        not related to any educational purposes. 

        For example, if topics are not the same, like a java textbook is given, 
        however, the user asks about a physics question, state the following: This
        content is not related to the inputted textbook, please select another textbook
        and try again.

         

        Question: {question}

        Context: {context}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}



        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        print(question)
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])
        return result

# #Setup Summarizer pipeline
class Summarizer:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-4o', temperature=0.6)

        #Setup a prompt template
        template = """\
        You are an assistant for summarizing the textbook.

        If the user has given a topic or topics, make the summary more focused on those topics.

        Give a nice and detailed summary. 

        Make sure to include all main points given. There can be no points missed 
        out. The output limit is 5 sentences minimum to 15 sentences maximum. 
        Use the textbook given ONLY, and give every point that is important and
        summarize those. 

        If the user states a specific chapter, ONLY GIVE THEM THE SUMMARY OF THAT
        SPECIFIC CHAPTER. 

        ALWAYS DOUBLE CHECK YOUR RESPONSE AND BE ACCURATE. DON'T TALK ABOUT ANOTHER
        TOPIC, ONLY GIVE INFORMATION YOU OBTAINED FROM THAT SPECIFIC TOPIC.

        If no query is given from the user, SUMMARIZE THE WHOLE TEXTBOOK GIVEN.

        Instead of giving a general summary of a number of pages, write bullet 
        points in terms of notes that the user can take into their notebook. Make
        sure to address the main points of the summary and don't generalize 
        specific topics. Just give bullet point summaries.

        USE MARKDOWN FORMATTING FOR BULLET POINTS. 

        If user requests summary as notes or flashcards like, give them nice bullet
        point notes so they can write it down.

        Question: {question}

        Context: {context}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}


        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])
        return result

# #Setup Quiz pipeline
class QuizAI:
    def __init__(self):
        self.doc = "tutor_textbook.pdf"
        self.loader = PyPDFLoader(self.doc)

        # Load the document and store it in the 'data' variable
        self.data = self.loader.load_and_split()

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.data, embedding=self.embeddings,
                                 persist_directory=".")

        # Initialize a language model with ChatOpenAI
        self.llm = ChatOpenAI(model_name= 'gpt-4o', temperature=0.6)

        #Setup a prompt template
        template = """\
            You are an assistant for quizzing topics on the textbook.

         Read through the texbook and generate quiz questions. You have to create 
        10 different quiz questions. They may be on the same topic or different 
        topics. If the user has given a query, focus the topic on that query. If 
        no query has been provided, use the whole textbook and pick topics at random.

        Your response format will ALWAYS be this:
        *** 
        Query: (THE USER'S QUERY GOES HERE)
        Quiz Question: (YOUR GENERATED QUIZ QUESTION GOES HERE)
        Quiz Correct Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Incorrect Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Another Incorrect Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Final Tricky Incorrect Answer: (YOUR GENERATED QUIZ ANSWER GOES HERE)
        Quiz Answer Explanation: (YOUR GENERATED EXPLANATION GOES HERE)
        ***

        The parenthesis inside the format are where you plug in the parameters if
        given. Say a user gives a query of "Chapter 1". You plug in the parameter
        with *** Query: Chapter 1 ***. If no query is given, put in the query field,
        NO QUERY. 

        Query: {question}

        Context: {context}

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain_type_kwargs = {"prompt": prompt}


        # 1. Vectorstore-based retriever
        self.vectorstore_retriever = self.vectordb.as_retriever()

        # Initialize a RetrievalQA chain with the language model and vector database retriever
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever= self.vectorstore_retriever, chain_type_kwargs=chain_type_kwargs)
        self.chat_history = []  # Initialize chat history

    def update_chat_history(self, question, answer):
        self.chat_history.append({"question": question, "answer": answer})

    def build_combined_context(self):
        """Combine chat history and document context."""
        # Combine all previous chat history
        chat_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in self.chat_history])
        
        # Fetch relevant context from the vector store based on the current question
        if self.chat_history:
            current_question = self.chat_history[-1]['question']
            context_from_db = self.vectorstore_retriever.get_relevant_documents(current_question)
        else:
            context_from_db = self.vectorstore_retriever.get_relevant_documents("")

        # Convert the list of context documents into a string
        context_str = "\n".join([doc.page_content for doc in context_from_db])

        # Combine both chat history and the document context
        combined_context = f"Chat history:\n{chat_context}\n\nContext from the document:\n{context_str}"
        
        return combined_context


    def invoke(self, input_dict):
        question = input_dict.get("question")
        combined_context = self.build_combined_context()

        result = self.qa_chain.invoke({
            "query": question,
            "context": combined_context
        })

        self.update_chat_history(question, result['result'])
        print(result['result'])

        return result


from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import markdown
from bs4 import BeautifulSoup
import json
import time, threading,requests
from waitress import serve
import gunicorn

filepath = "./tutor_textbook.pdf"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif'}
QUIZGENERATED = False

from flask import Flask
app = Flask(__name__)

@app.route("/hello")
def helloworld():
    return "Hello World - Vedant1223!"
if __name__ == "__main__":
    app.run()

@app.route("/",methods=["GET"])
def index():
   return render_template("index.html")

@app.route("/tutor-ai", methods=["GET", "POST"])
def tutor_ai():
    global url_data, prompt_data

    if request.method == "POST":
        file_name = "tutor_textbook.pdf"
        url_data = request.form.get("url")

        if 'file' not in request.files and not url_data:
            print('No file uploaded!')
        else:
            file = request.files['file']
            file.save(filepath)
            print("File saved:", filepath)

        if url_data:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            }

            response = requests.get(url_data, headers=headers, stream=True)
            response.raise_for_status()

            with open(file_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded {file_name} successfully.")

        prompt_data = request.form.get("prompt")
        base_qa_pipeline = BaseQAPipeline()
        result = base_qa_pipeline.invoke({'question': prompt_data})
        print(result)

        return render_template("tutor-ai.html", result=result)

    return render_template("tutor-ai.html")


@app.route("/summarizer", methods=["GET", "POST"])
def summarizer():
    global url_data, prompt_data  # Access global variables

    if request.method == "POST":
        url_data = request.form.get("url")
        print("URL: ", url_data)
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        if (url_data != ""):
            subprocess.check_call("curl", url_data, ">", "tutor_textbook.pdf")
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        base_qa_pipeline = Summarizer()
        result = base_qa_pipeline.invoke({'question' : prompt_data})
        print(result)
        return render_template("summarizer.html", result=result)
    return render_template("summarizer.html")

@app.route('/how-it-works', methods=['GET'])
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/study-plan', methods=['GET', "POST"])
def generate_plan():
    if request.method == "POST":
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        generate_plan = GenerateStudyPlan()
        result = generate_plan.invoke({'question' : prompt_data})
        result['result'] = markdown.markdown(result['result'])
        print(result)
        return render_template("study-plan.html", result=result)

    return render_template("study-plan.html")

@app.route('/quiz', methods=['GET', "POST"])
def quiz_maker():
    if request.method == "POST":
        if 'file' not in request.files:
            print('No file uploaded!')
        else:
          file = request.files['file']
          file.save(filepath)
          print("File saved:", filepath)
        print("File: ",file)
        prompt_data = request.form.get("prompt")
        quiz = QuizAI()
        print(prompt_data)
        result = quiz.invoke({'question' : prompt_data})
        print(result)

        # Parse the result HTML to extract the question, answer, and explanation
        soup = BeautifulSoup(result['result'], 'html.parser')
        
        # Extracting relevant parts
        result_text = result['result']
        quizzes = result_text.split("***")
        parsed_quizzes = []
        quiz_filepath = "static/quiz-data.json"

        for quiz in quizzes:
            lines = [line.strip() for line in quiz.split("\n") if line.strip()]
            if len(lines) < 7:
                continue  # Skip invalid or incomplete entries
            try:
                # Extracting quiz data
                question = lines[1].split(": ", 1)[1]
                correct_answer = lines[2].split(": ", 1)[1]
                incorrect_answers = [
                    lines[3].split(": ", 1)[1],
                    lines[4].split(": ", 1)[1],
                    lines[5].split(": ", 1)[1],
                ]
                explanation = lines[6].split(": ", 1)[1]

                # Add parsed quiz to the list
                parsed_quizzes.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "options": [correct_answer] + incorrect_answers,
                    "explanation": explanation
                })
            except Exception as e:
                print(f"Error parsing quiz: {quiz}\nError: {e}")


            # Write to a JSON file
            with open(quiz_filepath, "w") as quiz_file:
                json.dump(parsed_quizzes, quiz_file, indent=4)

            print("Quiz data successfully parsed and saved to 'quiz-data.json'!")

        # Render the template with the file path passed as a parameter
        return redirect(url_for('generated_quiz'))

    return render_template("quiz.html")

@app.route('/generated-quiz', methods=['GET'])
def generated_quiz():
    return render_template('generated-quiz.html', quizfile="static/quiz-data.json")

# Below is an implementation of clearing the storage for the user. NOT IMPLEMENTED YET
clear_signal = False
def reset_clear_signal():
    global clear_signal
    time.sleep(120)  # Wait for 10 seconds
    clear_signal = False  # Reset to False

@app.route('/clear-localstorage', methods=['POST'])
def clear_localstorage():
    global clear_signal
    clear_signal = True  # Set signal to True
    
    # Start a separate thread to reset the signal after 10 seconds
    threading.Thread(target=reset_clear_signal, daemon=True).start()
    
    return jsonify({"clear": True})  # Respond immediately with True

@app.route('/get-clear-status', methods=['GET'])
def get_clear_status():
    return jsonify({"clear": clear_signal})  # Return the current clear status

@app.route('/tutor_textbook', methods=['GET'])
def tutor_textbook():
    return send_from_directory(os.getcwd(), 'tutor_textbook.pdf')

if __name__ == "__main__":
    host = '0.0.0.0'
    port = int(os.environ.get("PORT", 8080))  
    print(f"Server is running on {host}:{port}")
    app.run(host=host, port=port)
    print("Stopping server...") 
    # above code is for SERVER
    #below code right now is to debug
    # print("Server is running...")
    # app.run(port=8081,debug=True)
    # print("Stopping server...") 