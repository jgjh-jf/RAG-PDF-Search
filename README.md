# RAG PDF Search — RTX 3050 Optimized

A local, GPU-optimized pipeline for searching and answering questions over collections of PDF documents using Retrieval-Augmented Generation (RAG). Built with Streamlit, LangChain, HuggingFace, and ChromaDB, this project is designed for efficient document ingestion and fast, accurate search — even on consumer GPUs like the RTX 3050.

## Features

- **Streamlit interface** for easy PDF search and Q&A.
- **PDF ingestion and chunking** (supports multi-page PDFs).
- **Document embeddings** via HuggingFace and SentenceTransformers.
- **ChromaDB vectorstore** for fast similarity search.
- **RAG pipeline** combining retrieval and LLM generation.
- **GPU-friendly settings** that work well on RTX 3050 and similar.

## Project Structure

- `ppa.py` — Streamlit app for interactive Q&A over your PDFs.
- `ingest.py` — CLI script to ingest PDFs, chunk them, create embeddings, and persist them in ChromaDB.
- `constant.py` — Sample/test script for creating collections in ChromaDB.
- `docs/` — Should contain your PDF files for ingestion.
- `db/` — Directory for persistent ChromaDB storage.

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   ```

2. **Install dependencies**

   You’ll need Python 3.8+, PyTorch, Streamlit, LangChain, ChromaDB, Transformers, and PDFMiner.
   Install recommended packages:

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install streamlit torch transformers langchain chromadb sentence-transformers pdfminer.six
   ```

   Optional for HuggingFace embeddings:

   ```bash
   pip install langchain-huggingface
   ```

3. **Prepare PDFs**

   Place your PDF files in the `docs/` directory.

## Usage

### PDF Ingestion

Run the ingestion script to parse PDFs, split them into chunks, compute embeddings, and store everything in ChromaDB.

```bash
python ingest.py
```
- Outputs are stored in `db/`.

### Launch the Q&A App

Start the Streamlit web UI:

```bash
streamlit run ppa.py
```

Then enter questions in the textbox — the app will retrieve relevant PDF content and provide answers using the locally hosted LLM.

## Configuration

- **GEN_MODEL**: Generation model name (default: `google/flan-t5-small`). Change for larger/smaller models as needed.
- **EMB_MODEL**: Embedding model for text chunks (default: `all-MiniLM-L6-v2`).
- **CHROMA_DIR**: Directory for vector database storage (default: `db/`).

## Notes

- Optimized for GPU, but works on CPU as well.
- Handles PDF loading errors gracefully.
- Can ingest multiple PDFs; scalable to moderate document collections.

## License

Add your license here (MIT, Apache-2.0, etc.)

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

---

*Questions or issues? Please open an issue or contribute!*
