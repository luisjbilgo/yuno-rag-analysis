import chromadb
from langchain_huggingface import HuggingFaceEmbeddings # O langchain_community si usaste la otra
from langchain_chroma import Chroma

# --- CONFIGURACIÓN ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "yuno_knowledge_base"
MODEL_NAME = "intfloat/e5-base-v2"

def main():
    print("🕵️‍♂️ Iniciando verificación de ChromaDB...")

    # 1. Conectar al Cliente
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    # Ver cuántos documentos hay
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        print(f"✅ Colección encontrada: '{COLLECTION_NAME}'")
        print(f"📊 Total de vectores almacenados: {count}")
    except Exception as e:
        print(f"❌ Error: No se encuentra la colección. {e}")
        return

    # 2. Cargar el modelo de Embeddings (Necesario para convertir tu pregunta en números)
    print("🧠 Cargando modelo E5 para consulta...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 3. Conectar LangChain
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # 4. PRUEBA SEMÁNTICA
    # IMPORTANTE: Con E5, las consultas deben llevar el prefijo "query: "
    query_text = "query: How does the android SDK work?"
    
    print(f"\n🔍 Buscando: '{query_text}'")
    results = vector_store.similarity_search_with_score(query_text, k=3)

    print("\n--- RESULTADOS RECUPERADOS ---")
    for i, (doc, score) in enumerate(results):
        print(f"\nResult #{i+1} (Score: {score:.4f})")
        print(f"📂 Source: {doc.metadata.get('source')}")
        print(f"🏷️ Type: {doc.metadata.get('type')}")
        
        # Si es imagen, mostramos la ruta
        if doc.metadata.get('type') == 'image':
            print(f"🖼️ Image Path: {doc.metadata.get('image_path')}")
            print(f"📝 Description: {doc.page_content}")
        else:
            # Si es texto, mostramos un fragmento
            print(f"📄 Content snippet: {doc.page_content[:150]}...")

if __name__ == "__main__":
    main()
