# Chatbot Multi-Agente - CVs del equipo

Este proyecto implementa un **chatbot multiagente** que permite hacer preguntas sobre los CVs de un equipo, respondiendo **solo con la información de cada persona**.

## Características

- Soporta múltiples CVs en PDF.
- Cada CV se convierte en un “agente” independiente.
- Búsqueda semántica con Pinecone para encontrar los fragmentos relevantes.
- Respuestas generadas por OpenAI GPT, asegurando que cada agente responda solo con su propio contexto.
- Interfaz web con Streamlit.

## Instalación



1)Creación entorno virtual


python -m venv venv_agenteCV 
venv_agenteCV\Scripts\activate 

pip install -r requirements.txt


2) Configuración de variables de entorno

export OPENAI_API_KEY="tu_api_key_openai"
export PINECONE_API_KEY="tu_api_key_pinecone"

3) Ejecución

python -m streamlit run AgenteCV18.py


##Flujo del sistema

1)Carga y extracción de texto de los CVs.

2)Dividir texto en fragmentos y generar embeddings.

3)Subir embeddings a Pinecone.

4)Usuario ingresa preguntas.

5)Para cada persona:

Consultar Pinecone y obtener fragmentos relevantes.

Limpiar menciones de otros agentes.

Generar respuesta con GPT basada solo en el contexto de esa persona.

6)Mostrar respuestas por persona en la interfaz.

## Diagrama de flujo (texto)

[Inicio]
   |
   v
[Subida de CVs en PDF]
   |
   v
[Extracción de texto de cada PDF]
   |
   v
[Creación de Documentos con metadatos]
   |
   v
[División del texto en fragmentos (chunks)]
   |
   v
[Generación de embeddings para cada fragmento]
   |
   v
[Subida de vectores a Pinecone (índice cv-index)]
   |
   v
[Usuario ingresa pregunta en Streamlit]
   |
   v
[Identificación de agentes mencionados en la pregunta]
   |
   v
+---------------------------+
| Por cada agente:          |
| 1. Consultar Pinecone con |
|    embeddings de la       |
|    pregunta               |
| 2. Obtener top_k fragmentos|
| 3. Limpiar contexto       |
|    eliminando otras      |
|    personas               |
| 4. Generar respuesta con |
|    GPT-3.5-Turbo solo    |
|    con su contexto       |
+---------------------------+
   |
   v
[Mostrar respuestas por persona en Streamlit]
   |
   v
[Fin]

## Gráfico de Flujo


