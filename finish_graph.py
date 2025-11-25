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
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN ---
LLM_MODEL_NAME = "gemini-2.5-flash"
judge_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME, 
    temperature=0,
    request_timeout=60,
    max_retries=5
)

# Embeddings Locales (CPU)
print("🔧 Configurando Embeddings Locales (HuggingFace)...")
judge_embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def run_finish_line():
    print("🚀 INICIANDO FASE FINAL: GRAFOS Y UNIFICACIÓN")
    
    # 1. CARGAR DATOS GENERADOS
    gen_file = "temp_results_generation.json"
    if not os.path.exists(gen_file):
        print("❌ Error fatal: No encuentro temp_results_generation.json")
        return
    
    with open(gen_file, 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    ground_truths = data['ground_truths']
    
    # 2. CARGAR RESULTADO VECTORIAL (YA EXISTENTE)
    vec_file = "partial_results_vector.csv"
    if os.path.exists(vec_file):
        print(f"✅ Cargando resultados vectoriales guardados desde {vec_file}")
        df_vector = pd.read_csv(vec_file)
    else:
        print("⚠️ Advertencia: No encontré partial_results_vector.csv. El resultado final solo tendrá Grafos.")
        df_vector = pd.DataFrame()

    # 3. EVALUAR GRAFOS (LO QUE FALTA)
    print("\n🕸️ Evaluando Sistema de Grafos (Neo4j)...")
    
    valid_indices_g = [i for i, ans in enumerate(data['g_answers']) if ans != "Error"]
    
    if valid_indices_g:
        ds_graph = Dataset.from_dict({
            "question": [questions[i] for i in valid_indices_g],
            "answer": [data['g_answers'][i] for i in valid_indices_g],
            "contexts": [data['g_contexts'][i] for i in valid_indices_g],
            "ground_truth": [ground_truths[i] for i in valid_indices_g]
        })
        
        metrics = [
            context_precision, context_recall, context_entity_recall,
            answer_relevancy, faithfulness, NoiseSensitivity()
        ]
        
        # Max workers 1 para estabilidad
        run_config = RunConfig(max_workers=1, timeout=120)
        
        try:
            res_graph = evaluate(
                ds_graph, metrics=metrics, 
                llm=judge_llm, embeddings=judge_embeddings, 
                run_config=run_config
            )
            df_graph = res_graph.to_pandas()
            df_graph['System'] = 'Graph'
            print("✅ Grafos Evaluado Exitosamente.")
        except Exception as e:
            print(f"❌ Error evaluando Grafos: {e}")
            df_graph = pd.DataFrame()
    else:
        print("⚠️ No hay respuestas válidas de grafo para evaluar.")
        df_graph = pd.DataFrame()

    # 4. UNIFICAR Y EXPORTAR
    print("\n💾 Unificando resultados...")
    if not df_vector.empty or not df_graph.empty:
        final_df = pd.concat([df_vector, df_graph])
        csv_name = "comparativa_final_rag.csv"
        final_df.to_csv(csv_name, index=False)
        
        print("\n" + "="*50)
        print("🏆 RESULTADOS FINALES COMPLETOS")
        print("="*50)
        print(final_df.groupby('System').mean(numeric_only=True))
        print(f"\n✅ Archivo guardado: {csv_name}")
    else:
        print("❌ No se generaron datos.")

if __name__ == "__main__":
    run_finish_line()
