"""
app.py — updated to:
 - accept file uploads of any type
 - extract text for common formats (PDF, DOCX, TXT, images via OCR, audio via OpenAI)
 - upload original file to S3 (using Batch_Store.s3 and S3_BUCKET_NAME)
 - upload extracted text to S3 under extracted_texts/
 - integrate with your existing RAG pipeline and save_message

REQUIRED PACKAGES (pip):
 pip install PyPDF2 python-docx docx2txt pillow pytesseract
 pip install boto3 werkzeug python-magic-bin==0.4.27  # optional for Windows; or python-magic on *nix
 pip install openai
 pip install pdf2image         # optional, if you want better PDF->image OCR coverage (requires poppler)

Note: pytesseract requires Tesseract OCR binary installed on the host:
 - Ubuntu/Debian: sudo apt-get install tesseract-ocr
 - Mac: brew install tesseract
 - Windows: install from https://github.com/tesseract-ocr/tesseract/releases and add to PATH
"""

import os
import re
import tempfile
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from dotenv import load_dotenv

# Langchain / RAG imports (unchanged)
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

# Batch storage & S3 client (you provided this module)
import Batch_Store

# File/Text extraction libs
import PyPDF2
import docx2txt
from PIL import Image
import pytesseract

# Optional: use python-magic for MIME detection (works on most OSes)
try:
    import magic  # python-magic

    HAVE_MAGIC = True
except Exception:
    HAVE_MAGIC = False

# OpenAI for audio transcription
import openai

# Logging
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Flask app
app = Flask(__name__)
load_dotenv()

# Environment / API KEYS
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

LOG.info(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
LOG.info(f"PINECONE_ENV: {PINECONE_ENV}")
LOG.info(f"OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")

# Embeddings & RAG setup (kept same as before)
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


# -------------------------
# Helper: text extraction
# -------------------------
def extract_text_from_pdf(path):
    """Extract text from PDF using PyPDF2 (best-effort)."""
    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text).strip()
    except Exception as e:
        LOG.exception("PDF extraction failed: %s", e)
        return ""


def extract_text_from_docx(path):
    """Extract text from DOCX using docx2txt."""
    try:
        text = docx2txt.process(path)
        return (text or "").strip()
    except Exception as e:
        LOG.exception("DOCX extraction failed: %s", e)
        return ""


def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        LOG.exception("TXT read failed: %s", e)
        return ""


def extract_text_from_image(path):
    """Use pytesseract to OCR an image file."""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except Exception as e:
        LOG.exception("Image OCR failed: %s", e)
        return ""


def transcribe_audio_with_openai(path):
    """
    Transcribe audio using OpenAI's transcription API.
    Uses the same approach as previously but wrapped to return text.
    """
    try:
        with open(path, "rb") as audio_file:
            # The method below depends on openai client version.
            # We'll use the recommended file->transcriptions API if available.
            # The exact call may differ by openai package version.
            # This tries to call the modern API.
            transcript = openai.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
            # transcript may be dict-like
            if isinstance(transcript, dict) and "text" in transcript:
                return transcript["text"]
            else:
                # Some clients return an object with .text
                return getattr(transcript, "text", "") or ""
    except Exception as e:
        LOG.exception("Audio transcription error: %s", e)
        return ""


def detect_mimetype(path, filename):
    """Attempt to detect MIME type; fallback to extension."""
    if HAVE_MAGIC:
        try:
            m = magic.Magic(mime=True)
            return m.from_file(path)
        except Exception:
            pass
    # fallback:
    ext = os.path.splitext(filename)[1].lower()
    ext_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webm": "audio/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
    }
    return ext_map.get(ext, "application/octet-stream")


# -------------------------
# Helper: S3 upload using Batch_Store.s3
# -------------------------
def upload_to_s3_local_and_bucket(local_path, s3_key, content_type=None):
    """
    Upload local_path -> S3 using Batch_Store.s3 and S3_BUCKET_NAME.
    Returns True on success, False on failure.
    """
    try:
        with open(local_path, "rb") as f:
            Batch_Store.s3.put_object(
                Bucket=Batch_Store.S3_BUCKET_NAME,
                Key=s3_key,
                Body=f,
                ContentType=content_type or "application/octet-stream"
            )
        LOG.info("Uploaded to s3://%s/%s", Batch_Store.S3_BUCKET_NAME, s3_key)
        return True
    except Exception as e:
        LOG.exception("S3 upload failed: %s", e)
        return False


# -------------------------
# Utilities
# -------------------------
def safe_filename_with_ts(original_filename):
    base = secure_filename(original_filename)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    name, ext = os.path.splitext(base)
    return f"{name}_{ts}{ext}"


# -------------------------
# Existing helper to clean LLM responses
# -------------------------
def clean_llm_response(response_dict):
    if isinstance(response_dict, dict) and "answer" in response_dict:
        response = response_dict["answer"]
        response = response.lstrip("?, ")
        response = re.sub(r"^\W+", "", response)
        return response
    else:
        return ""


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    """
    Handles:
      - form field 'msg'
      - file field 'file' (single file)
      - audio field 'audio' (webm blob)
    Behavior:
      - Save original file to temp and upload to s3 under uploads/
      - Extract text (if possible) and upload extracted text to s3 under extracted_texts/
      - For audio, do transcription using OpenAI and also upload audio to s3
      - Use extracted/transcribed text as `msg` for RAG processing
    """
    # Read inputs
    msg = request.form.get("msg", "").strip()
    uploaded_file = request.files.get("file")
    uploaded_audio = request.files.get("audio")

    LOG.info("Incoming request | msg present: %s | file: %s | audio: %s",
             bool(msg), getattr(uploaded_file, "filename", None), getattr(uploaded_audio, "filename", None))

    # Where we will store extracted text for processing
    extracted_text = ""

    # If audio was uploaded, save locally, upload to s3, transcribe (prefer transcription over msg)
    if uploaded_audio:
        safe_name = safe_filename_with_ts(getattr(uploaded_audio, "filename", "voice_message.webm"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_name)[1]) as tmp:
            uploaded_audio.save(tmp.name)
            local_audio_path = tmp.name

        # Upload raw audio to S3: uploads/<filename>
        s3_key_audio = f"uploads/{safe_name}"
        upload_to_s3_local_and_bucket(local_audio_path, s3_key_audio,
                                      content_type=detect_mimetype(local_audio_path, safe_name))

        # Transcribe
        try:
            transcript_text = transcribe_audio_with_openai(local_audio_path)
            extracted_text = transcript_text or ""
            LOG.info("Audio transcribed: %s", extracted_text[:200])
        except Exception as e:
            LOG.exception("Audio transcription failed: %s", e)
            extracted_text = ""

        # cleanup local file
        try:
            os.remove(local_audio_path)
        except Exception:
            pass

    # If a file was uploaded, save, upload to s3, and attempt to extract text based on extension/mime
    if uploaded_file:
        original_filename = getattr(uploaded_file, "filename", "uploaded_file")
        safe_name = safe_filename_with_ts(original_filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_name)[1]) as tmp:
            uploaded_file.save(tmp.name)
            local_path = tmp.name

        # Upload raw file to S3
        s3_key_file = f"uploads/{safe_name}"
        upload_ok = upload_to_s3_local_and_bucket(local_path, s3_key_file,
                                                  content_type=detect_mimetype(local_path, safe_name))

        # Try to extract text if file type is known
        mimetype = detect_mimetype(local_path, safe_name)
        LOG.info("Detected mimetype: %s for %s", mimetype, safe_name)

        # Branch by mimetype / extension
        ext = os.path.splitext(safe_name)[1].lower()
        try:
            if mimetype == "application/pdf" or ext == ".pdf":
                extracted_text = extract_text_from_pdf(local_path) or extracted_text

            elif mimetype in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",) or ext in (
            ".docx", ".doc"):
                extracted_text = extract_text_from_docx(local_path) or extracted_text

            elif mimetype.startswith("text") or ext == ".txt":
                extracted_text = extract_text_from_txt(local_path) or extracted_text

            elif mimetype.startswith("image") or ext in (".png", ".jpg", ".jpeg", ".gif", ".tiff", ".bmp"):
                extracted_text = extract_text_from_image(local_path) or extracted_text

            elif mimetype.startswith("audio") or ext in (".wav", ".mp3", ".webm", ".m4a", ".ogg"):
                # For uploaded audio files, attempt OpenAI transcription
                extracted_text = transcribe_audio_with_openai(local_path) or extracted_text

            else:
                # Unknown binary types — we won't attempt extraction; set a friendly message
                if not extracted_text:
                    extracted_text = ""

        except Exception as e:
            LOG.exception("Error during extraction for %s: %s", safe_name, e)
            extracted_text = extracted_text or ""

        # Upload extracted text to s3 if we got anything
        if extracted_text:
            txt_name = os.path.splitext(safe_name)[0] + ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp_txt:
                tmp_txt.write(extracted_text)
                tmp_txt_path = tmp_txt.name

            s3_key_text = f"extracted_texts/{txt_name}"
            upload_to_s3_local_and_bucket(tmp_txt_path, s3_key_text, content_type="text/plain; charset=utf-8")

            # cleanup tmp extracted file
            try:
                os.remove(tmp_txt_path)
            except Exception:
                pass

        # cleanup uploaded file
        try:
            os.remove(local_path)
        except Exception:
            pass

    # If there was audio-only transcription and no file, we already have extracted_text
    # If there was no file/audio but user sent text, use that
    if not extracted_text and msg:
        # Use user typed message
        used_msg = msg
    elif extracted_text:
        # prefer extracted text (audio->transcription or file-extracted)
        used_msg = extracted_text
    else:
        used_msg = ""

    if not used_msg:
        return "I did not receive any message or could not extract text from the uploaded file."

    # -------------------------
    # RAG + Response logic (unchanged)
    # -------------------------
    LOG.info("Processing message for RAG pipeline (first 200 chars): %s", used_msg[:200])

    # Predefined responses (same logic as you had before)
    stop_words = ["nothing", "bye", "stop", "exit", "thank you"]
    greetings = ["hello", "hi", "greetings", "hey", "what's up"]
    general_questions = ["how are you", "how are you doing", "how do you feel", "are you okay", "are you fine"]

    def is_pure_greeting(text):
        return text.lower().strip() in greetings

    # Decide how to respond
    if any(word in used_msg.lower() for word in stop_words):
        response = "Okay, have a great day! Goodbye!"

    elif is_pure_greeting(used_msg):
        response = "Hi, How can I help you ?"

    elif any(q in used_msg.lower() for q in general_questions):
        response = "I am an AI assistant and cannot feel emotions, but I am functioning properly. How can I assist you?"

    elif any(q in used_msg.lower() for q in ["where", "what", "when", "who", "how", "which", "?"]):
        result = rag_chain.invoke({"input": used_msg})
        LOG.info("Raw LLM Response: %s", str(result)[:1000])
        response = clean_llm_response(result) or str(result)

    else:
        result = rag_chain.invoke({"input": used_msg})
        LOG.info("Raw LLM Response: %s", str(result)[:1000])
        response = clean_llm_response(result) or str(result)

    # Save message pair using existing Batch_Store save_message (it adds to buffer + flushes)
    try:
        Batch_Store.save_message(used_msg, response)
    except Exception as e:
        LOG.exception("Error calling Batch_Store.save_message: %s", e)

    LOG.info("Final Response (first 200 chars): %s", response[:200])
    return str(response)


# -------------------------
# Run app
# -------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
