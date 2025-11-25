# 🧠 Yuno Knowledge Architect: RAG Vectorial vs. Knowledge Graph

![Python](https://img.shields.io/badge/Python-3.12-blue)
![RAGAS](https://img.shields.io/badge/AI-RAGAS_Evaluation-orange)
![Neo4j](https://img.shields.io/badge/DB-Neo4j_Graph-blue)
![ChromaDB](https://img.shields.io/badge/DB-Chroma_Vector-green)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-purple)

Este proyecto implementa y compara dos arquitecturas avanzadas de **Retrieval-Augmented Generation (RAG)** para responder preguntas técnicas sobre la documentación de la Fintech **Yuno**.

El sistema incluye un pipeline multimodal (Texto + Imágenes), despliegue en AWS EC2 y una interfaz interactiva en Streamlit para auditoría en tiempo real.

---

## 🏗️ Arquitectura del Sistema

Se diseñaron dos flujos de recuperación paralelos para evaluar cuál paradigma se adapta mejor a la documentación técnica.

### 1. Pipeline de Ingesta y Multimodalidad
* **Scraping:** Crawler personalizado que convierte HTML a **Markdown**, preservando la jerarquía semántica.
* **Visión (Patrón 3):** Las imágenes de la documentación se procesaron con **BLIP (Salesforce)** para generar descripciones textuales (*captions*) que permiten buscar diagramas mediante texto natural.

### 2. Estrategias RAG Comparadas

| Característica | 🔵 Arquitectura Vectorial (ChromaDB) | 🟠 Arquitectura de Grafos (Neo4j) |
| :--- | :--- | :--- |
| **Modelo de Datos** | Embeddings Densos (`intfloat/e5-base-v2`) | Grafo de Conocimiento (Nodos y Relaciones) |
| **Enriquecimiento** | **Metadata Injection:** Inyección de *topics* dentro del chunk de texto. | **NLP Determinístico:** Extracción de entidades con `KeyBERT`. |
| **Recuperación** | Similitud de Coseno (k=5). | Consulta Cypher basada en coincidencia de keywords. |
| **Generación** | Gemini 2.5 Flash. | Gemini 2.5 Flash. |

---

## 📊 Análisis Comparativo (Resultados RAGAS)

El sistema fue auditado utilizando el framework **RAGAS** con un dataset de control (*Ground Truth*) y **GPT-4o-mini** como juez imparcial.

### Hallazgos Principales

| Métrica | Vectorial | Grafos | Conclusión |
| :--- | :---: | :---: | :--- |
| **Context Recall** | **67%** | 30% | El modelo vectorial es superior encontrando información gracias a la similitud semántica (sinónimos). |
| **Faithfulness** | **76%** | 52% | Al recuperar contextos más ricos, el modelo vectorial alucina menos. |
| **Noise Sensitivity** | 29% | **9%** | **El Grafo es más limpio.** Si no encuentra la conexión exacta, no trae información irrelevante. |

### Veredicto Técnico
La arquitectura **Vectorial Enriquecida** resultó ser la más robusta para este caso de uso general. Sin embargo, el **Grafo** demostró una precisión "quirúrgica" ideal para validaciones estrictas donde el ruido es inaceptable. La recomendación para producción es una **Arquitectura Híbrida**: Vector para *Recall* y Grafo para *Re-ranking*.

---

## 🚀 Instalación y Uso

### Prerrequisitos
* Python 3.10+
* Instancia de Neo4j corriendo (Local o AuraDB)
* API Key de Google Gemini

### Pasos
1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/yuno-rag-project.git](https://github.com/tu-usuario/yuno-rag-project.git)
    cd yuno-rag-project
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar variables de entorno:**
    Crea un archivo `.env`:
    ```ini
    GOOGLE_API_KEY=tu_api_key
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=tu_password
    ```

4.  **Ejecutar la Web App:**
    ```bash
    streamlit run app.py
    ```

---

## 📱 Capturas de Pantalla

### Dashboard Comparativo
<img width="1563" height="966" alt="image" src="https://github.com/user-attachments/assets/636bdca1-aea4-4bba-a7ee-c735a430fa58" />


### Live Arena (Chat)
*<img width="1684" height="847" alt="image" src="https://github.com/user-attachments/assets/f0140f37-e0e9-4a0a-9922-871894d6e65a" />*

---

## 👨‍💻 Autor
Proyecto desarrollado para la materia de NLP y RAGs.
**[Luis Bilbao]** - [luisjbilgo2004@gmail.com]
