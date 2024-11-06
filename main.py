from fastapi import FastAPI, UploadFile, File
# , HTTPException
from sentence_transformers import SentenceTransformer
from utils.chroma_helper import add_document_to_chroma, search_chroma  # Import ChromaDB helper functions
from typing import List

app = FastAPI()

# Initialize the sentence-transformers model for embedding generation
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define path for document storage
DOCUMENT_STORE_PATH = "document_store/"

@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    for file in files:
        # Read file content
        contents = await file.read()
        
        # Generate embedding using the model
        embedding = model.encode(contents.decode('utf-8'))
        
        # Store document and embedding in ChromaDB
        add_document_to_chroma(file.filename, embedding)
        
        # Optional: Save file in document store folder
        file_path = DOCUMENT_STORE_PATH + file.filename
        with open(file_path, "wb") as f:
            f.write(contents)

    return {"message": f"Uploaded {len(files)} files successfully"}

@app.get("/query")
async def query_document(query: str):
    # Convert query to an embedding
    query_embedding = model.encode(query)
    
    # Query ChromaDB for similar documents
    results = search_chroma(query_embedding)
    return {"results": results}