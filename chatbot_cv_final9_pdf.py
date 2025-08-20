"""
Chatbot RAG para CVs en PDF usando Pinecone + OpenAI
Adaptado para ejecutarse en PC con Streamlit y PyTorch
"""

import os
import fitz  # PyMuPDF
import pinecone
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text
import streamlit as st


# CONFIGURACIÓN DE CLAVES


import os
import pinecone
from openai import OpenAI

# Cargar claves desde variables de entorno
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Inicializar clientes
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")  
client = OpenAI(api_key=openai_api_key)


# Nombre del índice
index_name = "cv-index"

# Crear índice si no existe
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="euclidean",
        serverless=True  # versión moderna equivalente a ServerlessSpec
    )

# Usar el índice
index = pc.Index(index_name)

# ===============================
# FUNCIONES
# ===============================
def extract_text_from_pdf(file_path):
    """Extrae texto de un PDF"""
    return extract_text(file_path)

def create_document(text, metadata=None):
    """Crea un objeto Document de LangChain"""
    if metadata is None:
        metadata = {}
    return Document(page_content=text, metadata=metadata)

def chunk_text(document, chunk_size=1000, chunk_overlap=200):
    """Divide el documento en chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents([document])

# Modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def semantic_search(query, top_k=3):
    query_vector = get_embedding(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    retrieved_texts = [item['metadata']['text'] for item in results['matches']]
    return retrieved_texts

def generate_answer(question):
    """Genera respuesta usando OpenAI y manejo de cuota limitada"""
    context_chunks = semantic_search(question)
    context = "\n\n".join(context_chunks)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # modelo más económico
            messages=[
                {"role": "system", "content": "Eres un asistente que responde usando la información proporcionada."},
                {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {question}"}
            ],
            max_tokens=300
        )
        return response.choices[0].message['content']
    except openai.RateLimitError:
        return "⚠️ Has excedido tu cuota de OpenAI. Espera a que se renueve o revisa tu plan."

# ===============================
# INTERFAZ STREAMLIT
# ===============================
st.title("📄 Chatbot RAG - CVs en PDF")

uploaded_file = st.file_uploader("Sube un CV en PDF", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    full_text = extract_text_from_pdf("temp.pdf")
    doc = create_document(full_text, metadata={"source": "temp.pdf"})
    chunks = chunk_text(doc)

    # Convertir a vectores y subir a Pinecone
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = {
            "id": f"chunk-{i}",
            "values": get_embedding(chunk.page_content),
            "metadata": {"text": chunk.page_content}
        }
        vectors.append(vector)

    index.upsert(vectors=vectors)
    st.success("✅ CV procesado y subido a Pinecone")

    # Chat
    user_question = st.text_input("Haz tu pregunta sobre el CV:")
    if user_question:
        with st.spinner("Buscando y generando respuesta..."):
            answer = generate_answer(user_question)
        st.markdown("**Respuesta:**")
        st.write(answer)
