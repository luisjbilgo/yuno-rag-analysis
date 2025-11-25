import json
import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall, context_entity_recall,
    answer_relevancy, faithfulness, NoiseSensitivity
)
from ragas.run_config import RunConfig 
# CAMBIO: Usamos OpenAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE PAGO (OPENAI) ---
# gpt-4o-mini es MUY barato ($0.15 / 1M tokens) y perfecto para evaluar.
print("💰 Configurando Juez OpenAI (gpt-4o-mini)...")
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Embeddings Locales (Gratis - Ahorramos tokens de OpenAI)
print("🔧 Configurando Embeddings Locales (HuggingFace)...")
judge_embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def run_fast_track():
    print("🚀 INICIANDO EVALUACIÓN RÁPIDA (OPENAI TRACK)")
    
    # 1. CARGAR DATOS GENERADOS
    gen_file = "temp_results_generation.json"
    if not os.path.exists(gen_file):
        print("❌ No encuentro 'temp_results_generation.json'. Ejecuta primero la generación.")
        return
    
    with open(gen_file, 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    ground_truths = data['ground_truths']
    
    # Preparamos índices válidos
    valid_indices_v = [i for i, ans in enumerate(data['v_answers']) if ans != "Error"]
    valid_indices_g = [i for i, ans in enumerate(data['g_answers']) if ans != "Error"]

    # Métricas
    metrics = [
        context_precision, context_recall, context_entity_recall,
        answer_relevancy, faithfulness, NoiseSensitivity()
    ]
    
    # CONFIGURACIÓN TURBO
    # Con OpenAI pagado podemos subir los workers.
    # Ponemos 4 para que vuele, pero sin exceder rate limits básicos.
    run_config = RunConfig(max_workers=4, timeout=120)

    print("\n🏎️  Evaluando Vectorial (Chroma)...")
    df_vector = pd.DataFrame()
    if valid_indices_v:
        ds_vector = Dataset.from_dict({
            "question": [questions[i] for i in valid_indices_v],
            "answer": [data['v_answers'][i] for i in valid_indices_v],
            "contexts": [data['v_contexts'][i] for i in valid_indices_v],
            "ground_truth": [ground_truths[i] for i in valid_indices_v]
        })
        res_vector = evaluate(ds_vector, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=run_config)
        df_vector = res_vector.to_pandas()
        df_vector['System'] = 'Vectorial'
        print("✅ Vectorial Listo.")

    print("\n🏎️  Evaluando Grafos (Neo4j)...")
    df_graph = pd.DataFrame()
    if valid_indices_g:
        ds_graph = Dataset.from_dict({
            "question": [questions[i] for i in valid_indices_g],
            "answer": [data['g_answers'][i] for i in valid_indices_g],
            "contexts": [data['g_contexts'][i] for i in valid_indices_g],
            "ground_truth": [ground_truths[i] for i in valid_indices_g]
        })
        res_graph = evaluate(ds_graph, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=run_config)
        df_graph = res_graph.to_pandas()
        df_graph['System'] = 'Graph'
        print("✅ Grafos Listo.")

    # EXPORTAR
    if not df_vector.empty or not df_graph.empty:
        final_df = pd.concat([df_vector, df_graph])
        final_df.to_csv("comparativa_final_openai.csv", index=False)
        
        print("\n" + "="*50)
        print("🏆 RESULTADOS FINALES (JUEZ: GPT-4o-MINI)")
        print("="*50)
        print(final_df.groupby('System').mean(numeric_only=True))
        print("\n✅ Archivo guardado: comparativa_final_openai.csv")

if __name__ == "__main__":
    run_fast_track()
