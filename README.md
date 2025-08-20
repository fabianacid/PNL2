# Chatbot RAG sobre CV (Pinecone + OpenAI)

Este proyecto implementa un chatbot que responde preguntas sobre el CV en PDF que se carga usando **Retrieval-Augmented Generation (RAG)**.


  
El sistema combina:  

- **Pinecone** → almacenamiento vectorial de embeddings.  
- **OpenAI GPT-3.5** → generación de respuestas con contexto.  
- **SentenceTransformers** → creación de embeddings de texto.  
- **Streamlit** → interfaz gráfica sencilla.  

---

##  Flujo del sistema

1. **Carga del PDF**  
   - El usuario sube un CV en formato PDF desde la interfaz de Streamlit.  
   - Se extrae todo el texto con `pdfminer.six`.  

2. **Creación de documentos y división en chunks**  
   - El texto se guarda como un objeto `Document` de LangChain.  
   - Luego se divide en fragmentos de 1000 caracteres con solapamiento de 200.  

3. **Generación de embeddings**  
   - Cada fragmento se convierte en un embedding con el modelo `all-MiniLM-L6-v2`.  

4. **Indexación en Pinecone**  
   - Los embeddings y su texto asociado se suben a un índice en Pinecone.  
   - Si el índice no existe, el sistema lo crea automáticamente.  

5. **Búsqueda semántica**  
   - El usuario escribe una pregunta.  
   - El sistema genera un embedding de esa consulta y busca los fragmentos más relevantes en Pinecone.  

6. **Generación de respuesta (RAG)**  
   - Los fragmentos recuperados se pasan como contexto a **OpenAI GPT-3.5-Turbo**.  
   - El modelo genera una respuesta combinando la información del CV con la pregunta.  

7. **Interfaz en Streamlit**  
   - Muestra:  
     - Estado del índice en Pinecone.  
     - Los fragmentos recuperados (para depuración).  
     - La respuesta final para el usuario.  

---

## Requisitos

- Python **3.9+**  
- Librerías principales:  
  - `torch`  
  - `sentence-transformers`  
  - `streamlit`  
  - `openai`  
  - `pinecone`  
  - `pdfminer.six`  
  - `PyMuPDF`  
  - `langchain`  

---

##  Instalación

1. Crear y activar un entorno virtual  

   ```bash
   python -m venv venv_chatbot
  
   venv_chatbot\Scripts\activate      # Windows


## Requisitos
- Python 3.10+
- Claves:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`



 
## Ejecuciones en la consola de windows

pip install -r requirements.txt

setx PINECONE_API_KEY "tu_clave_de_pinecone"
setx OPENAI_API_KEY "tu_clave_de_openai"



python -m streamlit run chatbot_cv_final9_pdf.py


