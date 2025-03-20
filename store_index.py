from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import sentence_transformers
from pinecone import ServerlessSpec  # Import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone by creating an instance
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Load and process data
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Pinecone index setup
index_name = "medicalbot"

# Create the index (if it doesn't exist)
if index_name not in pc.list_indexes().names():  # Check if the index exists
    # Create the index with the spec argument
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(  # Specify the index specification
            cloud='aws',
            region='us-east-1'  # You can change the region to your preference
        )
    )
else:
    print(f"Index {index_name} already exists.")

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
