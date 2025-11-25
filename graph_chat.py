import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from keybert import KeyBERT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN ---
load_dotenv()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ProyectoYuno2025" # <--- TU CONTRASEÑA DE NEO4J
LLM_MODEL = "gemini-2.5-flash"

# Inicialización (Global)
print("🕸️ Inicializando RAG de Grafos (Neo4j + KeyBERT + Gemini)...")

# KeyBERT para entender la pregunta y mapearla al grafo
kw_model = KeyBERT()

# LLM para generar la respuesta final
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.0
)

prompt_template = ChatPromptTemplate.from_template("""
Eres un asistente experto en Yuno. La información a continuación proviene de un Grafo de Conocimiento.
Usa EXCLUSIVAMENTE este contexto para responder.

CONTEXTO DEL GRAFO:
{context}

PREGUNTA:
{question}
""")

def get_graph_context(query_topics):
    """Ejecuta Cypher para buscar documentos conectados a los tópicos"""
    if not query_topics:
        return []

    # Query Cypher: Busca tópicos que coincidan parcialmente y trae sus documentos
    cypher_query = """
    UNWIND $topics AS search_topic
    MATCH (t:Topic)
    WHERE toLower(t.name) CONTAINS toLower(search_topic)
    MATCH (d:Document)-[:MENTIONS]->(t)
    RETURN DISTINCT d.title as Title, d.content as Content, d.url as URL
    LIMIT 5
    """
    
    contexts = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(cypher_query, topics=query_topics)
            for record in result:
                # Formateamos el resultado del grafo como texto
                ctx = f"Title: {record['Title']}\nContent: {record['Content']}\nSource: {record['URL']}"
                contexts.append(ctx)
        driver.close()
    except Exception as e:
        print(f"⚠️ Error conectando a Neo4j: {e}")
    
    return contexts

def get_graph_response(query: str):
    """
    Función principal llamada por el Benchmark.
    Retorna: (respuesta_texto, lista_de_contextos)
    """
    try:
        # 1. Extracción de Keywords (Mapping Pregunta -> Grafo)
        # Extraemos 2-gramas (ej: "credit card") para ser más precisos
        keywords = kw_model.extract_keywords(
            query, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=3
        )
        # Obtenemos lista limpia: ['android sdk', 'payment']
        search_topics = [k[0] for k in keywords]
        
        # 2. Retrieval (Graph Traversal)
        list_contexts = get_graph_context(search_topics)
        
        # Fallback: Si el grafo no retorna nada (keywords no coinciden)
        if not list_contexts:
            return "No encontré información relacionada en el Grafo de Conocimiento.", []

        # 3. Generation
        context_str = "\n\n".join(list_contexts)
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"context": context_str, "question": query})
        
        return response, list_contexts

    except Exception as e:
        print(f"❌ Error en Graph RAG: {e}")
        return "Error generando respuesta.", []
