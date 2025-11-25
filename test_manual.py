import time
import sys

# Agregamos el directorio actual al path para asegurar que encuentre los módulos
sys.path.append('.')

print("🚀 INICIANDO PRUEBA MANUAL DE INTEGRACIÓN...\n")

try:
    # --- PRUEBA 1: RAG VECTORIAL (Chroma) ---
    print("🔹 Cargando RAG Vectorial (rag_chat.py)...")
    from rag_chat import get_vector_response
    
    pregunta_v = "How do I install the iOS SDK with CocoaPods?"
    print(f"   ❓ Preguntando al Vector: '{pregunta_v}'")
    
    start = time.time()
    respuesta_v, contextos_v = get_vector_response(pregunta_v)
    end = time.time()
    
    print(f"   ⏱️ Tiempo: {end - start:.2f}s")
    if respuesta_v and "CocoaPods" in respuesta_v:
        print("   ✅ RESPUESTA VECTORIAL EXITOSA:")
        print(f"      Invoked: {respuesta_v[:100]}...") # Mostramos el inicio
        print(f"      Chunks recuperados: {len(contextos_v)}")
    else:
        print(f"   ⚠️ RESPUESTA DUDOSA: {respuesta_v}")

except ImportError as e:
    print(f"   ❌ Error importando rag_chat: {e}")
except Exception as e:
    print(f"   ❌ Error ejecutando rag_chat: {e}")

print("-" * 40)

try:
    # --- PRUEBA 2: RAG GRAFO (Neo4j) ---
    print("🔹 Cargando RAG de Grafos (graph_chat.py)...")
    from graph_chat import get_graph_response
    
    pregunta_g = "What are the payment methods supported by the Web SDK?"
    print(f"   ❓ Preguntando al Grafo: '{pregunta_g}'")
    
    start = time.time()
    respuesta_g, contextos_g = get_graph_response(pregunta_g)
    end = time.time()
    
    print(f"   ⏱️ Tiempo: {end - start:.2f}s")
    
    # Verificamos si trajo algo
    if contextos_g:
        print("   ✅ RESPUESTA GRAFO EXITOSA:")
        print(f"      Invoked: {respuesta_g[:100]}...")
        print(f"      Nodos recuperados: {len(contextos_g)}")
    else:
        print("   ⚠️ EL GRAFO NO ENCONTRÓ CONEXIONES (Keywords no matchearon).")
        print(f"      Respuesta: {respuesta_g}")

except ImportError as e:
    print(f"   ❌ Error importando graph_chat: {e}")
except Exception as e:
    print(f"   ❌ Error ejecutando graph_chat: {e}")

print("\n🏁 PRUEBA FINALIZADA")
