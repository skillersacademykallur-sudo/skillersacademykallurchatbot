import os
import tempfile
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv

# Langchain / RAG imports
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

import batch_store  # Custom module for saving data
from embedding import run_embedding_job
from datetime import datetime
from flask import request, jsonify
import threading

# File/Text libs
import PyPDF2
import docx2txt
from PIL import Image
import pytesseract
import openai

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Global In-Memory Chat History
chat_sessions = {}

# Init RAG
embeddings = download_hugging_face_embeddings()
index_name = "skillersbot12112025"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- Helpers (Text Extraction) ---
def extract_text_from_pdf(path):
    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t: text.append(t)
        return "\n".join(text).strip()
    except:
        return ""


def extract_text_from_docx(path):
    try:
        return (docx2txt.process(path) or "").strip()
    except:
        return ""


def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except:
        return ""


def extract_text_from_image(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img).strip()
    except:
        return ""


def transcribe_audio_with_openai(path):
    try:
        with open(path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="gpt-4o-transcribe", file=audio_file)
        return transcript.get("text", "")
    except:
        return ""


@app.route("/")
def index():
    return render_template("chat.html")


# -------------------------------------------------------
# REGISTRATION
# -------------------------------------------------------
@app.route("/register-user", methods=["POST"])
def register_user():
    try:
        user_id = str(uuid.uuid4())
        name = request.form.get("name")
        phone = request.form.get("phone")
        email = request.form.get("email")
        city = request.form.get("city")
        lat = request.form.get("lat")
        lon = request.form.get("lon")
        device = request.headers.get("User-Agent")

        # Save Profile
        user_data = {
            "user_id": user_id,
            "name": name,
            "phone": phone,
            "email": email,
            "location_city": city,
            "coordinates": {"lat": lat, "lon": lon},
            "device": device,
            "registered_at": datetime.utcnow().isoformat()
        }
        batch_store.save_user_info(user_data)

        # Init Session
        chat_sessions[user_id] = []

        resp = make_response(jsonify({"status": "ok", "message": "User registered"}))

        # Set Cookies (30 Days)
        max_age = 30 * 24 * 60 * 60
        resp.set_cookie("user_id", user_id, max_age=max_age)
        resp.set_cookie("user_name", name, max_age=max_age)
        resp.set_cookie("user_phone", phone, max_age=max_age)
        resp.set_cookie("user_city", city, max_age=max_age)
        resp.set_cookie("user_lat", str(lat), max_age=max_age)
        resp.set_cookie("user_lon", str(lon), max_age=max_age)

        return resp
    except Exception as e:
        LOG.error(f"Registration failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# NEW: SERVICE FORM SUBMISSION ROUTE
# -------------------------------------------------------
@app.route("/submit-service", methods=["POST"])
def submit_service():
    try:
        # 1. Extract Data from Form
        service_data = {
            "name": request.form.get("svcName"),
            "phone": request.form.get("svcPhone"),
            "address": request.form.get("svcAddress"),
            "state": request.form.get("svcState"),
            "district": request.form.get("svcDistrict"),
            "city": request.form.get("svcCity"),
            "service": request.form.get("svcService"),
            "details": request.form.get("svcDetails"),
            "user_id": request.cookies.get("user_id", "Anonymous"),
            "submitted_at": datetime.utcnow().isoformat()
        }

        # 2. Log and Save
        LOG.info(f"Service Request received: {service_data['service']} from {service_data['name']}")
        batch_store.save_service_request(service_data)
        # Run embedding job in background
        threading.Thread(target=run_embedding_job, daemon=True).start()

        return jsonify({"status": "success", "message": "Service details saved successfully."})


    except Exception as e:
        LOG.error(f"Service submission failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------
# CHAT
# -------------------------------------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    user_id = request.cookies.get("user_id", "Anonymous")

    user_name = request.cookies.get("user_name", "Anonymous")
    user_phone = request.cookies.get("user_phone", "")

    extracted_text = ""
    if 'file' in request.files:
        f = request.files['file']
        if f.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1]) as tmp:
                f.save(tmp.name)
                path = tmp.name
            ext = os.path.splitext(f.filename)[1].lower()
            if ext == ".pdf":
                extracted_text = extract_text_from_pdf(path)
            elif ext == ".docx":
                extracted_text = extract_text_from_docx(path)
            elif ext == ".txt":
                extracted_text = extract_text_from_txt(path)
            elif ext in [".jpg", ".jpeg", ".png"]:
                extracted_text = extract_text_from_image(path)
            os.remove(path)

    if 'audio' in request.files:
        a = request.files['audio']
        if a.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                a.save(tmp.name)
                path = tmp.name
            extracted_text = transcribe_audio_with_openai(path)
            os.remove(path)

    final_msg = msg
    if extracted_text:
        final_msg += f"\n[Attached File Content]: {extracted_text}"

    if not final_msg:
        return "I didn't receive any input."

    user_city = request.cookies.get("user_city", "Unknown City")
    user_lat = request.cookies.get("user_lat", "Unknown")
    user_lon = request.cookies.get("user_lon", "Unknown")

    context_info = f"USER DETAILS [Name: {user_name}, Location: {user_city}, Coords: {user_lat}, {user_lon}]. "

    history = chat_sessions.get(user_id, [])
    history_str = ""
    if history:
        history_str = "\nPREVIOUS CONVERSATION:\n" + "\n".join(
            [f"User: {h['user']}\nAI: {h['ai']}" for h in history]
        ) + "\n"

    full_input = f"{context_info}\n{history_str}\nCURRENT QUESTION: {final_msg}"

    result = rag_chain.invoke({"input": full_input})
    response = result.get("answer", "I'm sorry, I couldn't process that.")

    if user_id not in chat_sessions: chat_sessions[user_id] = []
    chat_sessions[user_id].append({"user": final_msg, "ai": response})
    if len(chat_sessions[user_id]) > 5: chat_sessions[user_id].pop(0)

    # Save Chat
    batch_store.save_message(user_id, user_name, user_phone, final_msg, response)

    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)