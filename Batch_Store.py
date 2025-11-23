import boto3
import json
import os
import threading
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import logging

# ---------------------------------------------------
# BASE DIRECTORY FOR STORING ALL LOG FILES
# ---------------------------------------------------
BASE_LOG_DIR = r"C:\Storage\DS\Projects\SkillersChatbotlogfiles"

# Standard Log Paths
LOG_DIR = BASE_LOG_DIR
CHAT_DIR = os.path.join(BASE_LOG_DIR, "daily_chats")
USER_DIR = os.path.join(BASE_LOG_DIR, "user_profiles")
SERVICE_DIR = os.path.join(BASE_LOG_DIR, "service_requests")

# RAG Data Path (For embedding.py)
RAG_DATA_DIR = r"C:\Storage\DS\Projects\skillersacademykallurchatbot\Data"

# Create directories if they don't exist
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)
os.makedirs(SERVICE_DIR, exist_ok=True)
os.makedirs(RAG_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------
# LOGGER CONFIGURATION
# ---------------------------------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

load_dotenv()

# ---------------------------------------------------
# S3 CLIENT
# ---------------------------------------------------
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


def upload_to_s3(local_path, s3_key):
    """Helper to upload file to S3 with retry logic."""
    if not S3_BUCKET_NAME:
        return

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
            logging.info(f"Synced to S3: {s3_key}")
            print(f"✅ Uploaded to S3: {s3_key}")
            return
        except ClientError as e:
            logging.error(f"S3 Upload Failed: {e}")
            if attempt < MAX_RETRIES - 1:
                threading._sleep(2)
        except Exception as e:
            logging.error(f"Unexpected S3 error: {e}")
            return


# ---------------------------------------------------
# 1. SAVE USER PROFILE
# ---------------------------------------------------
def save_user_info(user_data):
    try:
        safe_name = user_data.get("name", "Unknown").replace(" ", "_")
        safe_phone = user_data.get("phone", "NoNum")
        filename = f"Profile_{safe_name}_{safe_phone}.json"

        local_path = os.path.join(USER_DIR, filename)
        s3_key = f"user_profiles/{filename}"

        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2)

        if S3_BUCKET_NAME:
            threading.Thread(target=upload_to_s3, args=(local_path, s3_key)).start()

    except Exception as e:
        print(f"❌ Error Saving Profile: {e}")
        logging.error(f"Error saving user profile: {e}")


# ---------------------------------------------------
# 2. SAVE CHAT MESSAGE
# ---------------------------------------------------
def save_message(user_id, user_name, user_phone, user_msg, bot_response):
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        user_file_id = user_id.replace("-", "_")
        filename = f"Chat_{user_file_id}_{today}.json"

        local_path = os.path.join(CHAT_DIR, filename)
        s3_key = f"daily_chats/{filename}"

        current_data = []
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
            except json.JSONDecodeError:
                current_data = []

        new_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_msg": user_msg,
            "bot_response": bot_response,
            "user_name_at_time": user_name,
            "user_phone_at_time": user_phone
        }

        current_data.append(new_entry)

        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)

        if S3_BUCKET_NAME:
            threading.Thread(target=upload_to_s3, args=(local_path, s3_key)).start()

    except Exception as e:
        print(f"❌ Error Saving Chat: {e}")
        logging.error(f"Error saving chat message: {e}")


# ---------------------------------------------------
# 3. SAVE SERVICE REQUEST
# ---------------------------------------------------
def save_service_request(service_data):
    try:
        name = service_data.get("name", "Unknown").replace(" ", "_")
        service_type = service_data.get("service", "General").replace(" ", "_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        filename = f"Service_{service_type}_{name}_{timestamp}.json"

        if "submitted_at" not in service_data:
            service_data["submitted_at"] = datetime.utcnow().isoformat()

        local_path = os.path.join(SERVICE_DIR, filename)
        s3_key = f"service_requests/{filename}"

        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(service_data, f, indent=2)

        print(f"✅ Service Request Saved: {local_path}")
        logging.info(f"Service request saved locally: {filename}")

        # Copy to RAG folder
        rag_path = os.path.join(RAG_DATA_DIR, filename)
        with open(rag_path, "w", encoding="utf-8") as f:
            json.dump(service_data, f, indent=2)

        print(f"📤 Copied to RAG Data Folder: {rag_path}")

        # Sync to S3
        if S3_BUCKET_NAME:
            threading.Thread(target=upload_to_s3, args=(local_path, s3_key)).start()

    except Exception as e:
        print(f"❌ Error Saving Service Request: {e}")
        logging.error(f"Error saving service request: {e}")
