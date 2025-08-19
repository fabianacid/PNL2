"""
Chatbot RAG para CVs en PDF usando Pinecone + OpenAI
Adaptado para ejecutarse en PC con Streamlit
"""

import fitz  # PyMuPDF
import pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text
import streamlit as st

# ===============================
# CONFIGURACIÓN DE CLAVES
# ===============================
PINECONE_API_KEY = "TU_API_KEY_PINECONE"
OPENAI_API_KEY = "TU_API_KEY_OPENAI"

client = OpenAI(api_key=OPENAI_API_KEY)

# Inicializar Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")

index_name = "cv-index"

# Crear índice si no existe
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # debe coincidir con el modelo embeddings
        metric='euclidean'
    )

# Conectarse al índice
index = pinecone.Index(index_name)

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

# Modelo de embeddings local
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
    context_chunks = semantic_search(question)
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente que responde usando la información proporcionada."},
            {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {question}"}
        ]
    )
    return response.choices[0].message['content']

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
        vector = (
            f"chunk-{i}",
            get_embedding(chunk.page_content),
            {"text": chunk.page_content}
        )
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
