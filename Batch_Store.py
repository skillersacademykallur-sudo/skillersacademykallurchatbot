import boto3
import json
import threading
import atexit
from datetime import datetime
from flask import request
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# --- Load environment variables ---
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- Initialize S3 client ---
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# --- Global Message Buffer ---
message_buffer = []
buffer_lock = threading.Lock()
BATCH_SIZE = 10        # Flush after 10 messages
FLUSH_INTERVAL = 60    # Background flush interval in seconds

# --- Flush Function ---
def flush_to_s3():
    """Flush buffered messages to S3 safely."""
    global message_buffer
    with buffer_lock:
        if not message_buffer:
            return  # Nothing to write

        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        key = f"chat_batches/{timestamp}.json"

        # Copy and clear buffer
        batch = message_buffer.copy()
        message_buffer.clear()

    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(batch, indent=2),
            ContentType="application/json"
        )
        print(f"✅ Flushed {len(batch)} messages to s3://{S3_BUCKET_NAME}/{key}")

    except ClientError as e:
        print(f"[S3 Upload Error] {e}")
        # Re-add messages to buffer for next attempt
        with buffer_lock:
            message_buffer = batch + message_buffer

# --- Save Message ---
def save_message(user_msg, bot_response):
    """Add message + metadata to buffer, flush if needed."""
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

# --- Background Flush Thread ---
def periodic_flush():
    """Flush buffer periodically in the background."""
    flush_to_s3()
    timer = threading.Timer(FLUSH_INTERVAL, periodic_flush)
    timer.daemon = True
    timer.start()

# Start background flushing
periodic_flush()

# --- Final Flush on Shutdown ---
atexit.register(flush_to_s3)

print("✅ Batch flush system initialized and running in background...")
