import os
import chromadb

#client = chromadb.PersistentClient(path="./chroma_db")
#collection = client.get_or_create_collection("my_collection")
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection
collection = client.get_or_create_collection(name="my_collection")


collection.add(
    ids=["1", "2"],
    documents=["This is the first document.", "This is the second document."],
    metadatas=[{"source": "example1"}, {"source": "example2"}]
)

print("✅ ChromaDB collection created and sample data added successfully.")