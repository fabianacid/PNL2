import os
import time
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text
import streamlit as st

# -----------------------------
# CONFIGURACIÓN DE CLAVES
# -----------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "cv-index"

# Crear índice si no existe
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
AGENTE_ALUMNO = "maria fabiana cid"

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def get_embedding(text):
    return embedding_model.encode(text).tolist()

def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def create_document(text, metadata=None):
    if metadata is None:
        metadata = {}
    return Document(page_content=text, metadata=metadata)

def chunk_text(document, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents([document])

# -----------------------------
# FUNCIÓN CORREGIDA: RESPONDER POR PERSONA
# -----------------------------
def responder_pregunta_por_persona(personas, question, index, model="gpt-3.5-turbo"):
    respuestas = {}

    for persona in personas:
        # Buscar contexto SOLO de esa persona
        resultados = index.query(
            vector=get_embedding(question),
            filter={"source": {"$eq": persona}},  # Solo el CV de esta persona
            top_k=5,
            include_metadata=True
        )

        # Armar contexto
        context = "\n\n".join([res.metadata.get("text", "") for res in resultados.matches])

        if not context:
            respuestas[persona] = f"⚠️ No encontré información en el CV de {persona.title()}."
            continue

        # 🔹 Limpiar menciones de otras personas
        for otra_persona in personas:
            if otra_persona != persona:
                context = context.replace(otra_persona, "")

        # Pregunta personalizada
        pregunta_personalizada = (
            f"Responde únicamente sobre {persona.title()}. "
            "No incluyas información de otras personas. "
            f"Pregunta: {question}"
        )

        # Llamada al modelo
        try:
            time.sleep(1)  # evitar rate limit
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un asistente que responde solo en base al contexto entregado."},
                    {"role": "user", "content": f"Contexto:\n{context}\n\n{pregunta_personalizada}"}
                ],
                max_tokens=350
            )
            respuesta_texto = getattr(completion.choices[0].message, "content", "⚠️ No se pudo generar respuesta.")
            respuestas[persona] = respuesta_texto.strip()
        except Exception as e:
            respuestas[persona] = f"⚠️ Error generando respuesta: {str(e)}"

    return respuestas

# -----------------------------
# INTERFAZ STREAMLIT
# -----------------------------
st.title("📄 Chatbot Multi-Agente - CVs del equipo")

if "personas" not in st.session_state:
    st.session_state["personas"] = []

# Carga de CVs
uploaded_files = st.file_uploader("Sube uno o varios CVs en PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    vectors = []
    total_chunks = 0
    for uploaded_file in uploaded_files:
        persona = uploaded_file.name.replace(".pdf", "").replace("_", " ").strip().lower()
        if persona not in st.session_state["personas"]:
            st.session_state["personas"].append(persona)

        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        full_text = extract_text_from_pdf(file_path)
        doc = create_document(full_text, metadata={"source": persona})
        chunks = chunk_text(doc)

        for i, chunk in enumerate(chunks):
            vector = {
                "id": f"{persona}-chunk-{i}",
                "values": get_embedding(chunk.page_content),
                "metadata": {"text": chunk.page_content, "source": persona}
            }
            vectors.append(vector)

        total_chunks += len(chunks)
        st.success(f"✅ CV de **{persona.title()}** cargado correctamente con {len(chunks)} fragmentos.")

    # Subida a Pinecone
    time.sleep(3)
    index.upsert(vectors=vectors)
    st.info(f"📌 En total se subieron {total_chunks} fragmentos de texto a Pinecone.")

# Preguntas y respuestas
user_question = st.text_input("Haz tu pregunta sobre los CVs:")

if user_question:
    mencionados = []
    user_question_lower = user_question.lower()
    for persona in st.session_state["personas"]:
        if any(word in user_question_lower for word in persona.split()):
            mencionados.append(persona)
    if not mencionados:
        mencionados = [AGENTE_ALUMNO]

    respuestas = responder_pregunta_por_persona(mencionados, user_question, index)
    st.markdown("**Respuestas por persona:**")
    for persona, texto in respuestas.items():
        st.markdown(f"📌 **{persona.title()}:**")
        st.write(texto)
        st.markdown("---")
