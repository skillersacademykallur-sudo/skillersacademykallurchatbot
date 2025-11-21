from src.helper import load_files, text_split, download_hugging_face_embeddings
#from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from langchain.docstore.document import Document



load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENV: {PINECONE_ENV}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

print(os.getenv("\nPinecode api_key :, {PINECONE_API_KEY}"))


data_folder = "Data/"  # Change this to the correct folder path
extracted_data = load_files(data_folder)
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



pc_api_key = os.getenv('PINECONE_API_KEY')

print(os.getenv("\nPinecode api_key :, {pc_api_key}"))
pc = Pinecone(api_key=pc_api_key)

print(f"Pinecone Details: '{pc}' ")

#index_name = "medicalbot2" # commented on 12112025 to add excel embeddings
index_name = "skillersbot12112025"


try:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully!")
except Exception as e:
    print(f"Error creating index '{index_name}': {e}")


print(f"\nEmbeding each chunk and upsert the embeddings into Pinecone index.....")
# Embed each chunk and upsert the embeddings into your Pinecone index.
document = [Document(page_content=chunk) for chunk in text_chunks]
docsearch = PineconeVectorStore.from_documents(
    documents=document,
    index_name=index_name,
    embedding=embeddings, 
)
print(f"\nEmbeding each chunk and upsert the embeddings into Pinecone index is Done!\n")