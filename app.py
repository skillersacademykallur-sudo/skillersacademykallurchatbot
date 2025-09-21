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
from Batch_Store import save_message


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENV: {PINECONE_ENV}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")






embeddings = download_hugging_face_embeddings()


index_name = "skillersbot0925"

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
    msg = request.form.get("msg", "")
    print("User Message:", msg)

    stop_words = ["nothing", "bye", "stop", "exit", "thank you"]
    greetings = ["hello", "hi", "greeting", "hey", "what's up"]
    general_questions = ["how are you", "how are you doing", "how do you feel", "are you okay", "are you fine"]

    # --- Determine bot response ---
    if any(word in msg.lower() for word in stop_words):
        response = "Okay, have a great day! Goodbye!"
    elif any(greeting in msg.lower() for greeting in greetings):
        response = "Hi, How can I help you ?"
    elif not msg.isalpha() and len(msg) < 3:
        response = "I'm sorry, I could not understand that. Could you please type a proper question?"
    elif any(question in msg.lower() for question in general_questions):
        response = "I am an AI assistant and cannot feel emotions, but I am functioning properly. Thank you for asking. How can I assist you?"
    else:
        # Use RAG chain
        result = rag_chain.invoke({"input": msg})
        print("Raw LLM Response:", result)

        if isinstance(result, dict):
            response = clean_llm_response(result)
        elif isinstance(result, str):
            response = result.lstrip("?, ")
            response = re.sub(r"^\W+", "", response)
        else:
            response = "An error occurred."

    # --- Save user + bot message (metadata handled in Batch_Store) ---
    save_message(msg, response)

    print("Final Response:", response)
    return str(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
