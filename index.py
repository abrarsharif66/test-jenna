from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

per_dir="chroma_db"

def load_documents(directory: str) -> List:
    """Load PDF documents from a directory."""
    documents = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            pdf_loader = PyPDFLoader(pdf_path)
            documents.extend(pdf_loader.load())
    
    return documents

def chunk_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks: List, persist_directory: str) -> Chroma:
    """Create a Chroma vector store from document chunks."""
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    vector_store.persist()
    return vector_store

def load_vector_store(persist_directory: str) -> Chroma:
    """Load an existing Chroma vector store."""
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def similarity_search(vector_store: Chroma, query: str, k: int = 3) -> List:
    """Perform similarity search on the vector store."""
    results = vector_store.similarity_search(query, k=2)
    return results

dir_pdfs="C:/Users/Abrar sharif/VScodeprojects/jenna/pdfs"

loaded_docs=load_documents(dir_pdfs)
chunked=chunk_documents(loaded_docs)
create_vector_store(chunked,per_dir)

