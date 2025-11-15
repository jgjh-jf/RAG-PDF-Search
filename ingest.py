
import os
import sys
import traceback

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import Chroma


try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMB_CLASS = "huggingface"
except Exception:
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        EMB_CLASS = "sentence_transformer"
    except Exception:
        print("ERROR: No compatible embedding class installed. Install one of:")
        print("  pip install langchain-huggingface")
        print("or")
        print("  pip install langchain-community sentence-transformers")
        sys.exit(1)


def load_pdfs(folder="docs"):
    documents = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Loading: {pdf_path}")
                loader = PDFMinerLoader(pdf_path)
                try:
                    docs = loader.load()
                except Exception as e:
                    print(f"Failed to load {pdf_path}: {e}")
                    continue
                documents.extend(docs)
    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def get_texts_from_documents(docs):
   
    texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
   
    texts = [t for t in texts if t and t.strip()]
    return texts


def create_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    print(f"Using embedding class: {EMB_CLASS}, model: {model_name}")
    if EMB_CLASS == "huggingface":
        embedder = HuggingFaceEmbeddings(model_name=model_name)
    else:
        embedder = SentenceTransformerEmbeddings(model_name=model_name)

   
    try:
        vectors = embedder.embed_documents(texts)
    except Exception as e:
        print("Embedding computation failed:", e)
        traceback.print_exc()
        vectors = []

    return embedder, vectors


def main():
    # 1) Load PDFs
    documents = load_pdfs("docs")
    print("Total pages loaded:", len(documents))
    if len(documents) == 0:
        print("No PDFs/pages found in 'docs' folder. Please add PDFs and retry.")
        return

    # 2) Split into chunks
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
    print("Total chunks created:", len(chunks))
    if len(chunks) == 0:
        print("Text splitter returned zero chunks. Check document contents and splitter config.")
        return

    # 3) Convert Document objects to plain texts for embedding
    texts = get_texts_from_documents(chunks)
    print("Text items to embed:", len(texts))
    if len(texts) == 0:
        print("After filtering blank texts, no texts remain. Aborting.")
        return

    # 4) Create embeddings (and validate)
    embedder, vectors = create_embeddings(texts, model_name="all-MiniLM-L6-v2")
    print("Embeddings computed:", len(vectors))
    if not vectors or len(vectors) != len(texts):
        print("ERROR: Embedding vectors are empty or length mismatch.")
        print("This can be a network error (HuggingFace download), or embedding class failure.")
        return

    # 5) Persist to Chroma
    try:
        db = Chroma.from_documents(
    chunks,           # THEY ARE documents
    embedder,
    persist_directory="db"
     )

        # IMPORTANT: ensure persist() is called to save DB
        db.persist()
        print("Chroma DB saved to ./db")
    except Exception as e:
        print("Failed to create/persist Chroma DB:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()

