import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# === Configuración ===
st.set_page_config(page_title="Chat con tu CV (RAG)", page_icon="📄")

st.title("📄 Chatbot RAG sobre tu CV")
st.caption("Consulta tu CV con Pinecone + OpenAI")

# Variables de entorno necesarias
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "cv-index")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Faltan variables de entorno OPENAI_API_KEY y/o PINECONE_API_KEY")
    st.stop()

# === Inicializar embeddings y vectorstore ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# === Configurar LLM ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === Cadena Retrieval + Generación ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# === Interfaz ===
query = st.text_input("Escribí tu pregunta sobre el CV:")
if st.button("Preguntar") and query:
    with st.spinner("Buscando en el CV..."):
        result = qa_chain({"query": query})

    st.subheader("Respuesta")
    st.write(result["result"])

    st.subheader("Fuentes (chunks del CV)")
    for i, doc in enumerate(result["source_documents"], start=1):
        st.markdown(f"**Fuente {i}:**\n\n{doc.page_content[:500]}...")
