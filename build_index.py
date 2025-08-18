import argparse
from pathlib import Path
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

# === Inicializar Pinecone ===
def init_pinecone(api_key: str, index_name: str, dim: int = 1536):
    pc = Pinecone(api_key=api_key)
    if index_name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def extract_text_from_pdf(file_path: str) -> str:
    return extract_text(file_path)


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Ruta al PDF del CV")
    parser.add_argument("--index", default="cv-index", help="Nombre del índice Pinecone")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Falta variable de entorno PINECONE_API_KEY")

    print("Extrayendo texto...")
    text = extract_text_from_pdf(args.pdf)

    print("Aplicando chunking...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    # Embeddings OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Pinecone
    index = init_pinecone(api_key, args.index)
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=args.index
    )
    print(f"Chunks cargados en Pinecone index: {args.index}")
