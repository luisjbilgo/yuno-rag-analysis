import json
import time
import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall, context_entity_recall,
    answer_relevancy, faithfulness, NoiseSensitivity
)
from ragas.run_config import RunConfig 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings # <--- ESTO ES GRATIS (Corre en tu CPU)
from dotenv import load_dotenv

# Importamos tus módulos
from rag_chat import get_vector_response
from graph_chat import get_graph_response

load_dotenv()

# --- 1. CONFIGURACIÓN DEL JUEZ (GEMINI FREE TIER) ---
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    request_timeout=60,
    max_retries=3
)

# --- 2. CONFIGURACIÓN DE EMBEDDINGS (LOCALES / GRATIS) ---
# Esto NO cobra nada. Usa la CPU de tu EC2.
print("🔧 Configurando Embeddings Locales (HuggingFace - Gratis)...")
judge_embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def load_dataset():
    with open("evaluation_dataset.json", "r") as f:
        return json.load(f)

def safe_generate():
    """Genera respuestas y las guarda en JSON. Si falla, retoma desde el archivo."""
    raw_data = load_dataset()
    questions = [item["question"] for item in raw_data]
    ground_truths = [item["ground_truth"] for item in raw_data]
    
    output_file = "temp_results_generation.json"
    
    # Si ya existe el archivo, cargamos los datos previos
    if os.path.exists(output_file):
        print("📂 Encontré datos previos generados. Saltando generación...")
        with open(output_file, 'r') as f:
            return json.load(f)

    print("\n⚡ FASE 1: GENERANDO RESPUESTAS (Paso a Paso)...")
    
    v_answers, v_contexts = [], []
    g_answers, g_contexts = [], []

    # Procesamos preguntas una por una
    for i, q in enumerate(questions):
        print(f"   [{i+1}/{len(questions)}] Procesando: {q[:30]}...")
        
        # VECTOR RAG
        try:
            ans, ctx = get_vector_response(q)
            v_answers.append(ans if ans else "Error")
            v_contexts.append(ctx if ctx else ["No context"])
        except:
            v_answers.append("Error")
            v_contexts.append(["Error"])
        
        time.sleep(5) # Pausa de seguridad para Gemini Free Tier

        # GRAPH RAG
        try:
            ans, ctx = get_graph_response(q)
            g_answers.append(ans if ans else "Error")
            g_contexts.append(ctx if ctx else ["No context"])
        except:
            g_answers.append("Error")
            g_contexts.append(["Error"])
            
        time.sleep(5) # Pausa de seguridad

    # Guardamos todo en un JSON
    data = {
        "questions": questions,
        "ground_truths": ground_truths,
        "v_answers": v_answers, "v_contexts": v_contexts,
        "g_answers": g_answers, "g_contexts": g_contexts
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print("✅ Generación completada y guardada en disco.")
    return data

def run_evaluation():
    # 1. Obtener datos (ya sea generando ahora o cargando del archivo)
    data = safe_generate()
    
    # Extraemos listas
    questions = data['questions']
    ground_truths = data['ground_truths']
    
    # Filtramos respuestas fallidas para no evaluar errores
    valid_indices_v = [i for i, ans in enumerate(data['v_answers']) if ans != "Error"]
    valid_indices_g = [i for i, ans in enumerate(data['g_answers']) if ans != "Error"]

    print(f"\n👨‍⚖️ FASE 2: EVALUANDO CON RAGAS (Usando CPU Local para Embeddings)...")

    # Instanciamos métricas
    metrics = [
        context_precision, context_recall, context_entity_recall,
        answer_relevancy, faithfulness, NoiseSensitivity()
    ]
    
    # CONFIGURACIÓN LENTA PERO SEGURA
    # max_workers=1 evita que Gemini te bloquee por exceso de peticiones
    run_config = RunConfig(max_workers=1, timeout=120)

    # --- EVALUAR VECTOR ---
    df_vector = pd.DataFrame()
    if valid_indices_v:
        print("   ... Evaluando Vectorial (Chroma)")
        try:
            ds_vector = Dataset.from_dict({
                "question": [questions[i] for i in valid_indices_v],
                "answer": [data['v_answers'][i] for i in valid_indices_v],
                "contexts": [data['v_contexts'][i] for i in valid_indices_v],
                "ground_truth": [ground_truths[i] for i in valid_indices_v]
            })
            
            res_vector = evaluate(
                ds_vector, metrics=metrics, 
                llm=judge_llm, 
                embeddings=judge_embeddings, # <--- ESTO ES LO QUE TE AHORRA DINERO
                run_config=run_config
            )
            df_vector = res_vector.to_pandas()
            df_vector['System'] = 'Vectorial'
            print("   ✅ Vector Evaluado")
        except Exception as e:
            print(f"   ❌ Error evaluando Vector: {e}")

    # Pausa entre sistemas para dejar descansar la API
    time.sleep(10)

    # --- EVALUAR GRAPH ---
    df_graph = pd.DataFrame()
    if valid_indices_g:
        print("   ... Evaluando Grafos (Neo4j)")
        try:
            ds_graph = Dataset.from_dict({
                "question": [questions[i] for i in valid_indices_g],
                "answer": [data['g_answers'][i] for i in valid_indices_g],
                "contexts": [data['g_contexts'][i] for i in valid_indices_g],
                "ground_truth": [ground_truths[i] for i in valid_indices_g]
            })
            
            res_graph = evaluate(
                ds_graph, metrics=metrics, 
                llm=judge_llm, 
                embeddings=judge_embeddings, # <--- ESTO ES LO QUE TE AHORRA DINERO
                run_config=run_config
            )
            df_graph = res_graph.to_pandas()
            df_graph['System'] = 'Graph'
            print("   ✅ Graph Evaluado")
        except Exception as e:
            print(f"   ❌ Error evaluando Graph: {e}")

    # --- EXPORTAR ---
    if not df_vector.empty or not df_graph.empty:
        final_df = pd.concat([df_vector, df_graph])
        final_df.to_csv("comparativa_final_rag.csv", index=False)
        
        print("\n🏆 RESULTADOS FINALES:")
        # Imprimir solo las columnas numéricas promedio
        print(final_df.groupby('System').mean(numeric_only=True))
        print("\n✅ Archivo guardado: comparativa_final_rag.csv")
    else:
        print("⚠️ No se pudieron generar resultados.")

if __name__ == "__main__":
    run_evaluation()
