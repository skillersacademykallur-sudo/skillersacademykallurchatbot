import boto3
import json
import os
import threading
from datetime import datetime
from flask import request
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import logging

# --- Setup Logging ---
# Ensure these environment variables are set correctly, or use absolute paths
LOG_DIR = os.getenv("LOG_BASE_DIR", r"C:\Storage\DS\Projects\SkillersChatbotlogfiles")
CHAT_DIR = os.path.join(LOG_DIR, "daily_chats")
USER_DIR = os.path.join(LOG_DIR, "user_profiles")

# Create directories if they don't exist
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)

# Configure System Logger
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# Load environment variables (AWS credentials, bucket name, etc.)
load_dotenv()

# --- S3 Client ---
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Sanity check for S3 bucket name
if not S3_BUCKET_NAME:
    logging.warning("S3_BUCKET_NAME is not set in environment variables. S3 sync will fail.")


def upload_to_s3(local_path, s3_key):
    """Helper to upload file to S3 with retry logic."""
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
            logging.info(f"Synced to S3: {s3_key}")
            return
        except ClientError as e:
            logging.error(f"S3 Upload Failed for {s3_key} (Attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                # Simple backoff logic (e.g., wait 2 seconds before retrying)
                threading._sleep(2)
            else:
                logging.error(f"Failed to upload {s3_key} to S3 after {MAX_RETRIES} attempts.")
        except Exception as e:
            logging.error(f"Unexpected error during S3 upload for {s3_key}: {e}")
            return


# --- 1. SAVE USER PROFILE (One time per registration) ---
def save_user_info(user_data):
    """
    Saves user registration details to a specific file:
    Filename: Profile_{Name}_{Phone}.json
    """
    try:
        # Construct Filename based on name and phone (assumed to be unique at registration)
        safe_name = user_data.get("name", "Unknown").replace(" ", "_")
        safe_phone = user_data.get("phone", "NoNum")
        filename = f"Profile_{safe_name}_{safe_phone}.json"

        local_path = os.path.join(USER_DIR, filename)
        s3_key = f"user_profiles/{filename}"

        # Write Local
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2)

        # Upload S3 (Threaded to not block UI)
        if S3_BUCKET_NAME:
            threading.Thread(target=upload_to_s3, args=(local_path, s3_key)).start()
        else:
            logging.warning(f"S3 sync skipped for {filename} because bucket name is missing.")

        logging.info(f"User profile saved: {filename}")

    except Exception as e:
        logging.error(f"Error saving user profile: {e}")


# --- 2. SAVE CHAT MESSAGE (Appends to Daily Session File) ---
def save_message(user_id, user_name, user_phone, user_msg, bot_response):
    """
    Reads existing daily file, appends message, saves back.
    Filename: Chat_{user_id}_{YYYY-MM-DD}.json

    The critical fix is to use the immutable 'user_id' as the main identifier
    to ensure consistency when retrieving 'existing chats'.
    """
    try:
        # 1. Construct Unique Daily Filename
        today = datetime.utcnow().strftime("%Y-%m-%d")
        safe_name = user_name.replace(" ", "_")

        # --- FIX APPLIED HERE ---
        # Use the full, stable user_id to guarantee the same file path every time.
        # This prevents the creation of new files when 'user_phone' is missing
        # or inconsistent, which was the root cause of the bug.
        user_file_id = user_id.replace("-", "_")  # Sanitize the ID for the filename

        # New, robust filename based on the canonical user ID and date
        filename = f"Chat_{user_file_id}_{today}.json"

        local_path = os.path.join(CHAT_DIR, filename)
        s3_key = f"daily_chats/{filename}"

        # 2. Load Existing Data (if any)
        current_data = []
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
            except json.JSONDecodeError:
                # Handle corrupted or empty JSON file by starting a new list
                logging.warning(f"Corrupted or empty file detected at {local_path}. Starting new chat log.")
                current_data = []

        # 3. Append New Message
        new_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_msg": user_msg,
            "bot_response": bot_response,
            # Optionally include metadata for debugging
            "user_name_at_time": user_name,
            "user_phone_at_time": user_phone
        }
        current_data.append(new_entry)

        # 4. Write Back to Local
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)

        logging.info(f"Chat message appended locally to: {filename}")

        # 5. Sync to S3 (Threaded)
        if S3_BUCKET_NAME:
            threading.Thread(target=upload_to_s3, args=(local_path, s3_key)).start()
        else:
            logging.warning(f"S3 sync skipped for {filename} because bucket name is missing.")


    except Exception as e:
        logging.error(f"Error saving chat message: {e}")