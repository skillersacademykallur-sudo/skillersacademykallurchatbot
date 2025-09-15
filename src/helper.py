from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

print(f"\nHelper Mdoule Started")




def load_files(data_folder):
    extracted_texts = {}

    if not os.path.exists(data_folder):
        print(f"❌ Error: Folder '{data_folder}' not found!")
        return extracted_texts

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        text = ""

        print(f"\n📂 Processing: {filename}")

        try:
            # PDF
            if filename.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

                if not text.strip():
                    print(f"⚠️ No text in {filename}, attempting OCR...")
                    images = convert_from_path(file_path)
                    text = "\n".join(pytesseract.image_to_string(img) for img in images)

            # Word (.docx)
            elif filename.lower().endswith(".docx"):
                doc = Document(file_path)
                text = "\n".join(para.text for para in doc.paragraphs)

            # TXT
            elif filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            # CSV
            elif filename.lower().endswith(".csv"):
                df = pd.read_csv(file_path)
                text = df.to_string(index=False)

            else:
                print(f"ℹ️ Skipped unsupported file type: {filename}")
                continue

            # Save text file
            if text.strip():
                output_text_file = os.path.join(data_folder, os.path.splitext(filename)[0] + ".txt")
                with open(output_text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✅ Text extracted and saved to: {output_text_file}")
                extracted_texts[filename] = text
            else:
                print(f"⚠️ No text extracted from {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    return extracted_texts





#Split the Data into Text Chunks

def text_split(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_text(text)  # Use split_text() for raw text
    print("Length of Text Chunks:", len(text_chunks))
    # Optional: Print the first chunk for verification
    print("\nFirst Chunk:\n", text_chunks[0])
    return text_chunks


#Download the Embeddings from Hugging Face


def download_hugging_face_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    cache_folder: str = "./hf_cache"
) -> HuggingFaceEmbeddings:
    """
    Downloads Hugging Face embeddings and caches them locally.

    Args:
        model_name (str): Name of the Hugging Face embedding model.
        cache_folder (str): Path to store cached models.

    Returns:
        HuggingFaceEmbeddings: An embeddings object ready for use.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_folder,
        model_kwargs={"trust_remote_code": True}
    )

















# # Check if text extraction was successful
# if text:
#     # 2. Split text into chunks
#     text_chunks = text_split(text)

#     # 3. Download Hugging Face Embeddings
#     embeddings = download_hugging_face_embeddings()

#     # 4. Use embeddings (example usage)
#     if embeddings:
#         sample_text = text_chunks[0] if text_chunks else "No text available"
#         vector = embeddings.embed_query(sample_text)
#         print(f"\nSample Embedding Vector: {vector[:10]}")  # Print first 10 elements
#     else:
#         print("Error: Could not load embeddings.")
# else:
#     print("Error: No PDF text extracted.")
