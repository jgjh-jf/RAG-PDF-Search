import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import chroma
from langchain.chains import RetrievalQA

from langchain_community.llms import HuggingFacePipeline


# -------------------
# SETTINGS
# -------------------
GEN_MODEL = "google/flan-t5-small"   # small = SAFE for RTX 3050
EMB_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "db"

# -------------------
# LOAD MODEL SAFELY
# -------------------
@st.cache_resource
def load_generation_llm():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)

    # SAFE GPU LOAD
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            GEN_MODEL,
            torch_dtype=torch.float32   
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception as e:
        st.warning(f"GPU load failed: {e}. Falling back to CPU.")
        model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

   
    text_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=False,
        device=0 if torch.cuda.is_available() else -1
    )

    return HuggingFacePipeline(pipeline=text_pipe)


# BUILD RAG CHAIN

@st.cache_resource
def build_qa_chain():
    llm = load_generation_llm()
    embeddings = SentenceTransformerEmbeddings(model_name=EMB_MODEL)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    db = chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


def process_answer(question: str):
    qa = build_qa_chain()
    result = qa(question)
    return result.get("result"), result

# UI

def main():
    st.title("RAG PDF Search — RTX 3050 Optimized 🚀")

    q = st.text_area("Ask something about your PDFs:")
    if st.button("Ask") and q.strip():
        with st.spinner("Thinking..."):
            answer, meta = process_answer(q)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Metadata")
        st.write(meta)

if __name__ == "__main__":
    main()

