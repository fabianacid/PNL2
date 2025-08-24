# Chatbot Multi-Agente - CVs del equipo

Este proyecto implementa un **chatbot multiagente** que permite hacer preguntas sobre los CVs de un equipo, respondiendo **solo con la información de cada persona**.

## Características

- Soporta múltiples CVs en PDF.
- Cada CV se convierte en un “agente” independiente.
- Búsqueda semántica con Pinecone para encontrar los fragmentos relevantes.
- Respuestas generadas por OpenAI GPT, asegurando que cada agente responda solo con su propio contexto.
- Interfaz web con Streamlit.

## Instalación



1) Creación entorno virtual


python -m venv venv_agenteCV 

venv_agenteCV\Scripts\activate 

pip install -r requirements.txt


2) Configuración de variables de entorno

export OPENAI_API_KEY="tu_api_key_openai"

export PINECONE_API_KEY="tu_api_key_pinecone"

3) Ejecución

python -m streamlit run AgenteCV18.py


## Flujo del sistema

1) Carga y extracción de texto de los CVs.

2) Dividir texto en fragmentos y generar embeddings.

3) Subir embeddings a Pinecone.

4) Usuario ingresa preguntas.

5) Para cada persona:

Consultar Pinecone y obtener fragmentos relevantes.

Limpiar menciones de otros agentes.

Generar respuesta con GPT basada solo en el contexto de esa persona.

6) Mostrar respuestas por persona en la interfaz.

    

## Gráfico de Flujo

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/62b36a81-bb57-4903-9d6c-b80d4d97410f" />



