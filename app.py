import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Yuno Knowledge Architect",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem; 
        background: linear-gradient(to right, #1E88E5, #005cb2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
    }
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem;}
    .code-card {background-color: #1e1e1e; border-radius: 10px; padding: 10px; margin-bottom: 10px;}
    .success-box {padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; border-left: 5px solid #28a745;}
    .metric-box {background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- CARGA DE SISTEMAS RAG (Backend) ---
@st.cache_resource
def load_backend():
    try:
        from rag_chat import get_vector_response
        from graph_chat import get_graph_response
        return get_vector_response, get_graph_response, True
    except ImportError:
        return None, None, False

get_vector, get_graph, backend_ok = load_backend()

# --- SIDEBAR ---
with st.sidebar:
    # Si tienes el logo local o url
    st.image("https://docs.y.uno/img/logo.svg", width=180)
    st.markdown("### 🧭 Navegación")
    
    page = st.radio("Ir a:", [
        "1. Ingeniería de Datos",
        "2. Arquitecturas RAG",
        "3. Auditoría & Benchmarking",
        "4. Live Arena (Demo)"
    ])
    
    st.markdown("---")
    st.info("**Stack Tecnológico:**\n\n"
            "🕷️ **Scraping:** Bs4 + Markdownify\n"
            "👁️ **Visión:** BLIP (Image Captioning)\n"
            "⚡ **Vector DB:** Chroma + E5-Base\n"
            "🕸️ **Graph DB:** Neo4j + KeyBERT\n"
            "⚖️ **Juez:** GPT-4o-mini")

# --- HEADER ---
st.markdown("<div class='main-title'>🧠 Yuno Knowledge Architect</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Informe Técnico: Análisis Comparativo Vectorial vs. Grafos</div>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 1: INGENIERÍA DE DATOS (Scraping + BLIP)
# ==============================================================================
if page == "1. Ingeniería de Datos":
    st.header("🕷️ Fase 1: Recolección y Multimodalidad")
    
    st.markdown("""
    El primer desafío fue transformar la documentación técnica de Yuno (HTML) en una base de conocimiento estructurada y multimodal.
    """)

    tab1, tab2 = st.tabs(["A. Scraping & Limpieza", "B. Visión Computacional (BLIP)"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Estrategia de Scraping")
            st.info("""
            Se implementó un **Spider en Python** que recorre recursivamente `docs.y.uno`.
            
            **Decisión Clave de Diseño:**
            Convertir todo el HTML a **Markdown (`.md`)**.
            
            *¿Por qué?* El Markdown preserva la jerarquía semántica (Títulos `#`, Subtítulos `##`, Bloques de código ` ``` `). Esto permite que el chunking posterior respete los límites lógicos del documento.
            """)
        with col2:
            st.markdown("**Código del Scraper (Core Logic):**")
            st.code("""
def process_page(url):
    # ... (Requests & BeautifulSoup) ...
    
    # Encontrar contenedor principal
    content_div = soup.find('article') or soup.find('main')

    # Transformación a Markdown (Clave para RAG Técnico)
    markdown_content = md(str(content_div), heading_style="ATX")

    return {
        "url": url,
        "title": title,
        "content_markdown": markdown_content, # Contenido limpio
        "images": images_meta
    }
            """, language="python")

    with tab2:
        st.subheader("Implementación del Patrón Multimodal 3")
        st.markdown("""
        Las bases de datos vectoriales de texto no pueden indexar píxeles. Para solucionar esto, utilizamos el modelo **BLIP (Bootstrapping Language-Image Pre-training)**.
        
        El modelo "mira" cada imagen descargada y genera una descripción textual (`caption`) que luego se inyecta como metadato buscable.
        """)
        
        col_img1, col_img2 = st.columns([1.5, 1])
        
        with col_img1:
            st.code("""
# Procesamiento de Imágenes con HuggingFace
def generate_caption(processor, model, image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Inferencia en GPU/CPU
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_length=50)
    
    # Decodificación a texto natural
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
            """, language="python")
            
        with col_img2:
            st.success("""
            **Resultado Real:**
            - 142 Documentos procesados.
            - >100 Imágenes descritas.
            - JSONL enriquecido listo para ingesta.
            """)

# ==============================================================================
# PÁGINA 2: ARQUITECTURAS RAG
# ==============================================================================
elif page == "2. Arquitecturas RAG":
    st.header("🏗️ Fase 2: Diseño de Arquitecturas")
    
    st.markdown("Comparamos dos paradigmas de recuperación de información para responder preguntas técnicas.")

    col_v, col_g = st.columns(2)

    # --- COLUMNA VECTORIAL ---
    with col_v:
        st.markdown("### 🔵 Sistema Vectorial (ChromaDB)")
        st.markdown("Búsqueda por **Similitud Semántica**.")
        
        st.markdown("""
        ```mermaid
        graph TD
        A[Markdown Chunk] --> B(KeyBERT: Extraer Topics)
        B --> C{Metadata Injection}
        C --> D[Passage: [TOPICS] + Content]
        D --> E[Embedding E5-Base]
        E --> F[(ChromaDB)]
        ```
        """, unsafe_allow_html=True)
        
        with st.expander("Ver Código de Ingesta Vectorial"):
            st.code("""
# Ingesta Vectorial con Enriquecimiento
tags = extract_metadata_tags(kw_model, content)

# INYECCIÓN DE METADATOS
# Forzamos los tópicos dentro del texto vectorizable
passage_content = f"passage: [TOPICS: {tags}] {content}"

doc = Document(
    page_content=passage_content,
    metadata={"source": url, "type": "text"}
)
vector_store.add_documents([doc])
            """, language="python")

    # --- COLUMNA GRAFOS ---
    with col_g:
        st.markdown("### 🟠 Sistema de Grafos (Neo4j)")
        st.markdown("Búsqueda por **Conexiones Explícitas**.")
        
        st.markdown("""
        ```mermaid
        graph LR
        D((Documento)) -- MENTIONS --> T((Topic: 'Android'))
        D -- HAS_IMAGE --> I((Image))
        I -- DEPICTS --> T
        Q[Query Usuario] -.-> K{KeyBERT}
        K -.-> T
        ```
        """, unsafe_allow_html=True)
        
        with st.expander("Ver Código de Ingesta a Grafo"):
            st.code("""
# Construcción del Grafo en Neo4j
topics = extract_topics(kw_model, content)

cypher_query = \"""
MERGE (d:Document {url: $url})
WITH d
UNWIND $topics AS topic_name
MERGE (t:Topic {name: toLower(topic_name)})
MERGE (d)-[:MENTIONS]->(t)
\"""
session.run(cypher_query, url=url, topics=topics)
            """, language="python")

# ==============================================================================
# PÁGINA 3: AUDITORÍA & BENCHMARKING
# ==============================================================================
elif page == "3. Auditoría & Benchmarking":
    st.header("📊 Fase 3: Resultados del Benchmark (RAGAS)")
    
    csv_file = "comparativa_final_openai.csv"
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        st.markdown("""
        Se utilizó **GPT-4o-mini** como juez para evaluar 10 preguntas de control ("Ground Truth").
        """)

        # --- KPIs ---
        # Calculamos promedios agrupados
        means = df.groupby("System").mean(numeric_only=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Extraemos valores (Manejando nombres de columnas del CSV provisto)
        # Nombres en tu CSV: context_precision, context_recall, faithfulness, noise_sensitivity(mode=relevant)
        
        vec_prec = means.loc['Vectorial', 'context_precision']
        graph_prec = means.loc['Graph', 'context_precision']
        
        vec_recall = means.loc['Vectorial', 'context_recall']
        graph_recall = means.loc['Graph', 'context_recall']
        
        vec_faith = means.loc['Vectorial', 'faithfulness']
        graph_faith = means.loc['Graph', 'faithfulness']
        
        # Noise sensitivity
        ns_col = 'noise_sensitivity(mode=relevant)'
        vec_noise = means.loc['Vectorial', ns_col]
        graph_noise = means.loc['Graph', ns_col]

        col1.metric("Context Precision", f"{vec_prec:.1%}", delta=f"{(vec_prec-graph_prec)*100:.1f}% vs Graph")
        col2.metric("Context Recall", f"{vec_recall:.1%}", delta=f"{(vec_recall-graph_recall)*100:.1f}% vs Graph")
        col3.metric("Faithfulness", f"{vec_faith:.1%}", delta=f"{(vec_faith-graph_faith)*100:.1f}% vs Graph")
        # Noise sensitivity es mejor si es menor (invertimos color delta)
        col4.metric("Noise Sensitivity", f"{graph_noise:.1%}", delta=f"{(vec_noise-graph_noise)*100:.1f}% mejor", delta_color="normal")

        st.markdown("---")

        # --- GRÁFICOS ---
        c_chart1, c_chart2 = st.columns([2, 1])
        
        with c_chart1:
            st.subheader("Comparativa General de Métricas")
            # Transformar para Plotly
            df_melt = means.reset_index().melt(id_vars='System', var_name='Métrica', value_name='Score')
            # Limpiar nombres largos
            df_melt['Métrica'] = df_melt['Métrica'].replace('noise_sensitivity(mode=relevant)', 'Noise Sensitivity')
            
            fig = px.bar(df_melt, x='Métrica', y='Score', color='System', barmode='group',
                         color_discrete_map={'Vectorial': '#2E86C1', 'Graph': '#E67E22'},
                         height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with c_chart2:
            st.subheader("Análisis de Radar")
            categories = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[means.loc['Vectorial', c] for c in categories],
                theta=categories, fill='toself', name='Vectorial'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[means.loc['Graph', c] for c in categories],
                theta=categories, fill='toself', name='Graph'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- CONCLUSIONES ---
        st.markdown("""
        <div class='success-box'>
        <h3>🏆 Veredicto Técnico</h3>
        <p><b>Ganador Global: Arquitectura Vectorial Enriquecida.</b></p>
        <ul>
            <li><b>Recall (+122%):</b> El modelo E5 demostró ser superior entendiendo sinónimos ("pay" vs "transaction"). El Grafo falló sistemáticamente cuando la keyword del usuario no era exacta.</li>
            <li><b>Fidelidad (+44%):</b> Al recuperar chunks de texto completos, el vector proporciona mejor contexto al LLM para generar respuestas fieles.</li>
            <li><b>La Ventaja del Grafo:</b> Su <b>Noise Sensitivity (0.09)</b> es extremadamente baja. Esto significa que es "quirúrgico": o trae la respuesta exacta o no trae nada. Ideal para sistemas de tolerancia cero a la alucinación.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Ver Datos Crudos (CSV)"):
            st.dataframe(df)

    else:
        st.error("⚠️ No se encontró el archivo de resultados `comparativa_final_openai.csv`.")

# ==============================================================================
# PÁGINA 4: LIVE ARENA
# ==============================================================================
elif page == "4. Live Arena (Demo)":
    st.header("⚔️ La Arena: Vector vs Graph en Vivo")
    st.markdown("Prueba los sistemas en tiempo real para validar las métricas.")

    if not backend_ok:
        st.error("❌ Error de conexión con los servicios (Chroma/Neo4j). Revisa los logs.")
        st.stop()

    # Historial
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input
    query = st.chat_input("Pregunta sobre Yuno (Ej: What are the security requirements?)")
    
    if query:
        with st.spinner("🧠 Consultando ambos cerebros..."):
            # 1. Vector
            resp_v, ctx_v = get_vector(query)
            # 2. Grafo
            resp_g, ctx_g = get_graph(query)
            
            st.session_state.history.append({
                "q": query,
                "v": {"ans": resp_v, "ctx": ctx_v},
                "g": {"ans": resp_g, "ctx": ctx_g}
            })

    # Renderizado
    for chat in reversed(st.session_state.history):
        st.markdown(f"#### 👤: *{chat['q']}*")
        
        col_a, col_b = st.columns(2)
        
        # --- VECTOR ---
        with col_a:
            st.markdown("""<div class='code-card' style='border-top: 4px solid #2E86C1'>
                        <h4 style='color:#2E86C1'>🔵 Vectorial (Chroma)</h4>
                        </div>""", unsafe_allow_html=True)
            st.write(chat['v']['ans'])
            with st.expander("🔍 Ver Contexto Vectorial"):
                for c in chat['v']['ctx']:
                    st.text(f"• {c[:200]}...")

        # --- GRAFO ---
        with col_b:
            st.markdown("""<div class='code-card' style='border-top: 4px solid #E67E22'>
                        <h4 style='color:#E67E22'>🟠 Grafo (Neo4j)</h4>
                        </div>""", unsafe_allow_html=True)
            st.write(chat['g']['ans'])
            with st.expander("🕸️ Ver Nodos del Grafo"):
                if chat['g']['ctx']:
                    for c in chat['g']['ctx']:
                        st.text(f"• {c[:200]}...")
                else:
                    st.caption("⚠️ No se encontraron conexiones exactas.")
        
        st.divider()
