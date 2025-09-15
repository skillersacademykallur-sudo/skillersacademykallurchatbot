from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import re


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENV: {PINECONE_ENV}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")






embeddings = download_hugging_face_embeddings()


index_name = "skillersbot1"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatOpenAI(model="gpt-4o")
#llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



def clean_llm_response(response_dict):
    """Cleans the LLM response from a dictionary."""
    if isinstance(response_dict, dict) and "answer" in response_dict:
        response = response_dict["answer"]
        response = response.lstrip("?, ")
        response = re.sub(r"^\W+", "", response)
        return response
    else:
        return ""

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    stop_words = ["nothing", "bye", "stop", "exit", "thank you"]
    greetings = ["hello", "hi", "greeting", "hey", "what's up"]
    general_questions = ["how are you", "how are you doing", "how do you feel", "are you okay", "are you fine"]

    if any(word in msg.lower() for word in stop_words):
        return "Okay, have a great day! Goodbye!"

    if any(greeting in msg.lower() for greeting in greetings):
        return "Hi, How can I help you ?"

    if not msg.isalpha() and len(msg) < 3:  # very short or symbols
        return "I'm sorry, I could not understand that. Could you please type a proper question?"

    if any(question in msg.lower() for question in general_questions):
        return "I am an AI assistant and cannot feel emotions, but I am functioning properly. Thank you for asking. How can I assist you?"

    result = rag_chain.invoke({"input": msg}) # result is now defined.
    print("Raw LLM Response:", result)
    
    if isinstance(result, dict):
        response = clean_llm_response(rag_chain.invoke({"input": msg}))
        print("Raw LLM Response:", response) 
        print("Response : ", response)
        return str(response)
    elif isinstance(result, str):
        response = response.lstrip("?, ")
        response = re.sub(r"^\W+", "", response)
        print("Response : ", response)
        return str(response)
    else:
        return "An error occurred."





if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
