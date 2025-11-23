import os
import json
import logging
import threading
import atexit
from datetime import datetime

from flask import request
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# ==============================
# Logging Configuration
# ==============================
LOG_DIR = r"C:\Storage\DS\Projects\SkillersChatbotlogfiles"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# ==============================
# Initialize S3 Client
# ==============================
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# ==============================
# Batch Buffering Settings
# ==============================
message_buffer = []
buffer_lock = threading.Lock()

BATCH_SIZE = 10
FLUSH_INTERVAL = 60  # seconds

LOCAL_JSON_DIR = os.path.join(LOG_DIR, "chat_batches")
os.makedirs(LOCAL_JSON_DIR, exist_ok=True)

# ==============================
# Local Save Handler
# ==============================
def flush_to_local(batch):
    """Flush batch to local folder as JSON."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    file_path = os.path.join(LOCAL_JSON_DIR, f"{timestamp}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2)
        logging.info(f"Flushed {len(batch)} messages to {file_path}")
    except Exception as e:
        logging.error(f"Local JSON Save Error: {e}")

# ==============================
# S3 Save Handler
# ==============================
def flush_to_s3():
    """Flush buffered messages to S3 safely."""
    global message_buffer

    with buffer_lock:
        if not message_buffer:
            return

        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        key = f"chat_batches/{timestamp}.json"

        batch = message_buffer.copy()
        message_buffer.clear()

    flush_to_local(batch)

    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(batch, indent=2),
            ContentType="application/json"
        )
        logging.info(f"Flushed {len(batch)} messages to s3://{S3_BUCKET_NAME}/{key}")

    except ClientError as e:
        logging.error(f"S3 Upload Error: {e}")

        # Re-queue messages
        with buffer_lock:
            message_buffer = batch + message_buffer

# ==============================
# Save Message
# ==============================
def save_message(user_msg, bot_response):
    """Add message + metadata to the buffer and auto-flush if needed."""
    global message_buffer

    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_ip": request.remote_addr,
        "user_agent": request.headers.get("User-Agent"),
        "endpoint": request.path,
        "user_message": user_msg,
        "bot_response": bot_response,
    }

    with buffer_lock:
        message_buffer.append(data)
        if len(message_buffer) >= BATCH_SIZE:
            flush_to_s3()

# ==============================
# Background Periodic Flush
# ==============================
def periodic_flush():
    flush_to_s3()
    timer = threading.Timer(FLUSH_INTERVAL, periodic_flush)
    timer.daemon = True
    timer.start()

periodic_flush()

# ==============================
# Cleanup on Shutdown
# ==============================
atexit.register(flush_to_s3)

logging.info("Batch flush system initialized and running in background...")
