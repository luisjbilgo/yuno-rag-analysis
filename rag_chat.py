import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN ---
load_dotenv()

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
# Asegúrate de usar el nombre de la colección V2 que creamos con Topics
COLLECTION_NAME = "yuno_knowledge_base_enriched" 

# Modelos
EMBEDDING_MODEL = "intfloat/e5-base-v2"
LLM_MODEL = "gemini-2.5-flash"

# Inicialización de clientes (Global para no reconectar en cada pregunta)
print("🔌 Inicializando RAG Vectorial (Chroma + Gemini)...")

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # Traemos 5 chunks para tener buen contexto
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.0, # Temperatura 0 para evaluación objetiva
    max_output_tokens=1024
)

prompt_template = ChatPromptTemplate.from_template("""
Eres un asistente experto en la documentación de Yuno.
Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta.
Si la respuesta no está en el contexto, di "No tengo información suficiente en el contexto proporcionado".

CONTEXTO RECUPERADO:
{context}

PREGUNTA:
{question}
""")

def format_docs(docs):
    """Convierte los documentos recuperados en un solo string para el prompt"""
    formatted = []
    for doc in docs:
        content = doc.page_content.replace("passage: ", "") # Limpieza E5
        source = doc.metadata.get("title", "Untitled")
        formatted.append(f"--- Documento: {source} ---\n{content}")
    return "\n".join(formatted)

def get_vector_response(query: str):
    """
    Función principal llamada por el Benchmark.
    Retorna: (respuesta_texto, lista_de_contextos)
    """
    try:
        # 1. Retrieval (Añadimos prefijo query: para E5)
        search_query = f"query: {query}"
        retrieved_docs = retriever.invoke(search_query)
        
        # 2. Preparar Contexto para RAGAS (Lista de strings)
        list_contexts = [doc.page_content for doc in retrieved_docs]
        context_str = format_docs(retrieved_docs)
        
        # 3. Generation
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"context": context_str, "question": query})
        
        return response, list_contexts

    except Exception as e:
        print(f"❌ Error en Vector RAG: {e}")
        return "Error generando respuesta.", []
