import os
import json
import chromadb
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from keybert import KeyBERT # <--- LA JOYA PARA METADATOS

load_dotenv()

# --- CONFIGURACIÓN ---
DATA_PATH = "yuno_data/yuno_docs_captioned.jsonl"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "yuno_knowledge_base_enriched"

# Usamos E5 para los vectores principales (Alta precisión)
MODEL_NAME = "intfloat/e5-base-v2"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": True}

def extract_metadata_tags(kw_model, text):
    """
    Usa KeyBERT para extraer tópicos semánticos del texto.
    Retorna una cadena separada por comas (ej: "android sdk, payment flow, credit card")
    """
    try:
        # Extraemos 5 keywords, keyphrase_ngram_range=(1, 2) permite frases de 2 palabras
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=5
        )
        # KeyBERT devuelve una lista de tuplas [('palabra', 0.8), ...]
        # Nos quedamos solo con las palabras
        tags = ", ".join([k[0] for k in keywords])
        return tags
    except Exception as e:
        print(f"⚠️ Warning extrayendo tags: {e}")
        return ""

def load_data_enriched(kw_model):
    text_docs = []
    image_docs = []
    
    print(f"📂 Leyendo archivo: {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        return [], []

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        # Leemos todo primero para tener un contador real
        lines = f.readlines()
        total = len(lines)
        
        print(f"✨ Iniciando enriquecimiento de metadatos para {total} documentos...")
        
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
                source_url = record.get('url', '')
                title = record.get('title', 'Sin título')

                # --- 1. TEXTO ---
                if record.get('content_markdown'):
                    content = record['content_markdown']
                    
                    # ENRIQUECIMIENTO DE METADATOS 🧠
                    # Extraemos tags del contenido completo antes de chunkear
                    # (Esto da contexto global al documento)
                    tags = extract_metadata_tags(kw_model, content[:2000]) # Limitamos a 2000 chars por velocidad
                    
                    # Preparamos para E5
                    passage_content = f"passage: [TOPICS: {tags}] {content}"
                    
                    doc = Document(
                        page_content=passage_content,
                        metadata={
                            "source": source_url, 
                            "title": title, 
                            "type": "text",
                            "topics": tags # <--- AQUÍ ESTÁ EL VALOR AGREGADO
                        }
                    )
                    text_docs.append(doc)
                    
                    if i % 10 == 0:
                        print(f"   ... Procesando doc {i}/{total} | Tags detectados: [{tags}]")

                # --- 2. IMÁGENES ---
                if record.get('images'):
                    for img in record['images']:
                        caption = img.get('caption_blip')
                        if caption and len(caption) > 5:
                            # También sacamos tags de la descripción de la imagen
                            img_tags = extract_metadata_tags(kw_model, caption)
                            
                            passage_content = f"passage: Image Description: {caption}"
                            
                            img_doc = Document(
                                page_content=passage_content,
                                metadata={
                                    "source": source_url, 
                                    "title": title, 
                                    "type": "image",
                                    "image_path": img.get('local_path'),
                                    "topics": img_tags # <--- Tags visuales
                                }
                            )
                            image_docs.append(img_doc)
            except Exception as e:
                print(f"Error en línea {i}: {e}")
                continue
                
    return text_docs, image_docs

def main():
    # 1. Configurar Cliente Chroma
    print(f"🔌 Conectando a ChromaDB...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # 2. Inicializar Modelos
    print("🧠 Cargando modelo KeyBERT para extracción de metadatos...")
    kw_model = KeyBERT() # Utiliza un modelo MiniLM ligero por defecto
    
    print(f"🧠 Cargando modelo E5 ({MODEL_NAME}) para vectores...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS
    )

    # 3. Carga y Enriquecimiento
    raw_text, raw_images = load_data_enriched(kw_model)
    
    # 4. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )

    print("\n✂️  Creando chunks y propagando metadatos...")
    text_chunks = text_splitter.split_documents(raw_text)
    # Nota: Los chunks heredan automáticamente los metadatos 'topics' del documento padre
    
    all_docs = text_chunks + raw_images
    print(f"📦 Vectores finales: {len(all_docs)}")

    # 5. Ingesta
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    print("🚀 Insertando datos enriquecidos...")
    batch_size = 30 
    total = len(all_docs)
    
    for i in range(0, total, batch_size):
        batch = all_docs[i:i + batch_size]
        ids = [str(uuid4()) for _ in range(len(batch))]
        vector_store.add_documents(documents=batch, ids=ids)
        print(f"   ... Ingestando batch {i}/{total}")

    print("🎉 ¡INGESTA AVANZADA COMPLETADA!")
    print("Cada chunk ahora tiene palabras clave semánticas en sus metadatos.")

if __name__ == "__main__":
    main()
