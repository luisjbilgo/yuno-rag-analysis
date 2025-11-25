import json
import os
from neo4j import GraphDatabase
from keybert import KeyBERT
from dotenv import load_dotenv

# --- CONFIGURACIÓN ---
load_dotenv()

# Ajusta con TU contraseña que definiste en el paso anterior
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ProyectoYuno2025" 

DATA_PATH = "yuno_data/yuno_docs_captioned.jsonl"

def setup_database(driver):
    """Crea índices para que la inserción y búsqueda sean rápidas"""
    queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.path IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)"
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)
    print("✅ Base de datos configurada (Índices creados).")

def extract_topics(kw_model, text):
    """Usa KeyBERT para sacar keywords (Topics)"""
    try:
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=4 # 4 Tópicos por documento para no saturar el grafo
        )
        return [k[0] for k in keywords]
    except:
        return []

def ingest_graph_data():
    print(f"🔌 Conectando a Neo4j en {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Conexión exitosa.")
    except Exception as e:
        print(f"❌ Error conectando a Neo4j: {e}")
        return

    # 1. Preparar DB
    setup_database(driver)

    # 2. Cargar Modelos
    print("🧠 Cargando KeyBERT...")
    kw_model = KeyBERT()

    # 3. Leer Datos
    if not os.path.exists(DATA_PATH):
        print("❌ No se encuentra el archivo de datos.")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total = len(lines)
        print(f"🚀 Iniciando ingesta de {total} documentos al Grafo...")

        with driver.session() as session:
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                    url = record.get('url', '')
                    title = record.get('title', 'Sin Título')
                    content = record.get('content_markdown', '')[:3000] # Limitamos texto para no explotar RAM

                    if not content: continue

                    # A. Extraer Tópicos del Texto
                    topics = extract_topics(kw_model, content)

                    # B. CYPHER QUERY: Crear Documento y conectar con Tópicos
                    # Usamos MERGE para no duplicar si corres el script 2 veces
                    cypher_doc = """
                    MERGE (d:Document {url: $url})
                    SET d.title = $title, 
                        d.content = $content,
                        d.type = 'text'
                    
                    WITH d
                    UNWIND $topics AS topic_name
                    MERGE (t:Topic {name: toLower(topic_name)})
                    MERGE (d)-[:MENTIONS]->(t)
                    """
                    
                    session.run(cypher_doc, url=url, title=title, content=content, topics=topics)

                    # C. Manejo de Imágenes (Multimodal en Grafo)
                    if record.get('images'):
                        for img in record['images']:
                            caption = img.get('caption_blip')
                            path = img.get('local_path')
                            
                            if caption and len(caption) > 5:
                                # Extraer tópicos de la imagen también
                                img_topics = extract_topics(kw_model, caption)
                                
                                cypher_img = """
                                MATCH (d:Document {url: $url})
                                MERGE (i:Image {path: $path})
                                SET i.caption = $caption, i.type = 'image'
                                MERGE (d)-[:HAS_IMAGE]->(i)
                                
                                WITH i
                                UNWIND $img_topics AS t_name
                                MERGE (t:Topic {name: toLower(t_name)})
                                MERGE (i)-[:DEPICTS]->(t)
                                """
                                session.run(cypher_img, url=url, path=path, caption=caption, img_topics=img_topics)

                    if i % 10 == 0:
                        print(f"   ... Procesado {i}/{total} | Tópicos: {topics}")

                except Exception as e:
                    print(f"⚠️ Error en línea {i}: {e}")
                    continue

    driver.close()
    print("🎉 ¡Ingesta de GRAFO completada!")

if __name__ == "__main__":
    ingest_graph_data()
