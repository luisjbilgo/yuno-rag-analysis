from neo4j import GraphDatabase
from keybert import KeyBERT
from dotenv import load_dotenv
import os

# --- CONFIGURACIÓN ---
load_dotenv()
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ProyectoYuno2025"  # Tu contraseña

def main():
    print("🕵️‍♂️ Iniciando verificación de Grafo...")
    
    # 1. Conexión
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return

    # 2. Estadísticas Generales
    print("\n📊 ESTADÍSTICAS DEL GRAFO:")
    with driver.session() as session:
        result = session.run("""
            MATCH (n) 
            RETURN labels(n) as Type, count(n) as Count
            ORDER BY Count DESC
        """)
        for record in result:
            print(f"   • {record['Type'][0]}: {record['Count']} nodos")

    # 3. SIMULACIÓN DE RAG (Retrieval)
    # Imaginemos que el usuario pregunta esto:
    user_query = "How to customize the Android SDK theme?"
    print(f"\n🔍 Simulando búsqueda para: '{user_query}'")

    print("🧠 Extrayendo keywords de la pregunta con KeyBERT...")
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    search_topics = [k[0] for k in keywords]
    print(f"   🔑 Keywords detectadas: {search_topics}")

    # 4. Consulta Cypher (El corazón del Graph RAG)
    # Buscamos Tópicos que coincidan y traemos los Documentos conectados
    cypher_query = """
    UNWIND $topics AS search_topic
    MATCH (t:Topic)
    WHERE toLower(t.name) CONTAINS toLower(search_topic)
    MATCH (d:Document)-[:MENTIONS]->(t)
    RETURN d.title as Title, d.url as URL, t.name as MatchedTopic, d.content as Content
    LIMIT 3
    """

    print("\n--- RESULTADOS RECUPERADOS DEL GRAFO ---")
    with driver.session() as session:
        results = session.run(cypher_query, topics=search_topics)
        found = False
        for record in results:
            found = True
            print(f"\n📄 Documento: {record['Title']}")
            print(f"   🔗 Conectado por el tópico: [{record['MatchedTopic']}]")
            print(f"   🌐 URL: {record['URL']}")
            print(f"   📝 Snippet: {record['Content'][:100]}...")
        
        if not found:
            print("⚠️ No se encontraron conexiones exactas. (Esto es normal en pruebas pequeñas)")

    driver.close()

if __name__ == "__main__":
    main()
