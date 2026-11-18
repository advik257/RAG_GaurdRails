import glob
import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from config import Config

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]

def load_pdf(file_path):
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path, "page": i + 1}
            ))
    return docs

def load_documents(data_dir="data"):
    docs = []
    for file in glob.glob(f"{data_dir}/*"):
        if file.endswith(".txt"):
            docs.extend(load_txt(file))
        elif file.endswith(".pdf"):
            docs.extend(load_pdf(file))
    return docs

def ingest():
    print("Loading docs...")
    docs = load_documents()
    print(f"Found {len(docs)} docs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = CohereEmbeddings(
        model=Config.EMBEDDING_MODEL,
        cohere_api_key=Config.COHERE_API_KEY
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=Config.CHROMA_DIR,
        collection_name=Config.COLLECTION_NAME
    )
    print(f"Ingested successfully -> {Config.CHROMA_DIR}")

if __name__ == "__main__":
    ingest()