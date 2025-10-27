"""
tools_everlast.py
=================
Módulo que encapsula todas las herramientas para el agente inteligente.
Versión compatible - Lee vector store con manejo de errores Windows.

IMPORTANTE: Ejecutar primero: python codigo/create_vectorstore.py

Autor: Evaluación 2 - Everlast Chile
"""

import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
import requests

from langchain_core.tools import Tool


# =============================================================================
# CACHÉ EN MEMORIA
# =============================================================================

_vector_store_cache = None
_chunks_cache = None


# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

def obtener_vector_store():
    """
    Carga el vector store FAISS desde disco (versión compatible Windows).
    
    Returns:
        tuple: (faiss_index, chunks)
    
    Raises:
        FileNotFoundError: Si el vector store no existe
    """
    global _vector_store_cache, _chunks_cache
    
    # Retornar desde caché si existe
    if _vector_store_cache is not None and _chunks_cache is not None:
        print(">> Reutilizando Vector Store desde caché...")
        return _vector_store_cache, _chunks_cache
    
    print(">> Cargando Vector Store desde disco...")
    
    # Detectar ruta
    if Path.cwd().name == "codigo":
        vectorstore_path = Path("../datos/vectorstore_faiss")
    else:
        vectorstore_path = Path("datos/vectorstore_faiss")
    
    vectorstore_path = vectorstore_path.resolve()
    
    # Verificar que existe
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"❌ ERROR: Vector store no encontrado en: {vectorstore_path}\n"
            f"   Debes ejecutar primero: python codigo/create_vectorstore.py"
        )
    
    try:
        import faiss
        
        # Intentar cargar FAISS (compatible Windows)
        faiss_file = str(vectorstore_path / "index.faiss")
        
        # Verificar que el archivo existe
        if not Path(faiss_file).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {faiss_file}")
        
        # Cargar índice FAISS
        try:
            index = faiss.read_index(faiss_file)
        except Exception as e1:
            # Método alternativo para Windows
            print(f"   Intento 1 falló, probando método alternativo...")
            import shutil
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.faiss') as tmp:
                tmp_path = tmp.name
            shutil.copy(faiss_file, tmp_path)
            index = faiss.read_index(tmp_path)
            os.unlink(tmp_path)
        
        # Cargar chunks
        with open(vectorstore_path / "chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        print(f"   ✅ Vector Store cargado ({index.ntotal} vectores)")
        
        # Guardar en caché
        _vector_store_cache = index
        _chunks_cache = chunks
        
        return index, chunks
        
    except Exception as e:
        raise Exception(f"❌ ERROR al cargar vector store: {e}")


def buscar_similares(query: str, k: int = 3) -> List[str]:
    """
    Busca los k chunks más similares a la query usando HTTP directo.
    
    Args:
        query: Texto de búsqueda
        k: Número de resultados
    
    Returns:
        List[str]: Lista de textos relevantes
    """
    # Cargar vector store
    index, chunks = obtener_vector_store()
    
    # Generar embedding de la query usando HTTP
    github_token = os.environ.get("GITHUB_TOKEN")
    embeddings_url = os.environ.get("OPENAI_EMBEDDINGS_URL")
    
    if not github_token or not embeddings_url:
        raise ValueError("Variables de entorno GITHUB_TOKEN y OPENAI_EMBEDDINGS_URL requeridas")
    
    # Hacer request HTTP
    url = f"{embeddings_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": [query]
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Error al generar embedding: {response.status_code}")
    
    data = response.json()
    query_embedding = np.array([data['data'][0]['embedding']], dtype=np.float32)
    
    # Buscar en FAISS
    distances, indices = index.search(query_embedding, k)
    
    # Obtener chunks correspondientes
    resultados = []
    for idx in indices[0]:
        if idx < len(chunks):
            resultados.append(chunks[idx].page_content)
    
    return resultados


# =============================================================================
# DEFINICIÓN DE HERRAMIENTAS
# =============================================================================

def buscar_documentos_everlast(query: str) -> str:
    """
    Busca información relevante en la base de conocimiento de Everlast.
    
    Args:
        query: Pregunta o consulta del usuario
    
    Returns:
        str: Fragmentos relevantes encontrados o mensaje de error
    """
    print(f"\n🔍 [TOOL] BusquedaDocumentosEverlast")
    print(f"   Query: '{query}'")
    
    try:
        # Buscar chunks similares
        resultados = buscar_similares(query, k=3)
        
        if not resultados:
            mensaje = "No se encontró información relevante en los documentos de Everlast."
            print(f"   ⚠️  {mensaje}")
            return mensaje
        
        # Formatear contexto
        contexto = "\n\n--- FRAGMENTO ---\n\n".join(resultados)
        
        print(f"   ✅ {len(resultados)} fragmentos recuperados")
        print(f"   📄 Vista previa: {contexto[:150]}...")
        
        return contexto
        
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return str(e)
        
    except Exception as e:
        error_msg = f"Error al buscar en documentos: {str(e)}"
        print(f"   ❌ {error_msg}")
        return error_msg


def simple_calculator(expression: str) -> str:
    """
    Calculadora simple para operaciones matemáticas básicas.
    
    Args:
        expression: Expresión matemática como string
    
    Returns:
        str: Resultado del cálculo o mensaje de error
    """
    print(f"\n🔢 [TOOL] CalculadoraSimple")
    print(f"   Expresión: '{expression}'")
    
    try:
        # Validar caracteres
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            error_msg = "Expresión no válida. Solo números y operadores básicos (+, -, *, /, (), .)"
            print(f"   ❌ {error_msg}")
            return f"Error: {error_msg}"
        
        # Evaluar
        resultado = eval(expression)
        respuesta = f"El resultado de '{expression}' es: {resultado}"
        
        print(f"   ✅ {respuesta}")
        return respuesta
        
    except ZeroDivisionError:
        error_msg = "Error: División por cero no permitida."
        print(f"   ❌ {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Error al calcular '{expression}': {str(e)}"
        print(f"   ❌ {error_msg}")
        return error_msg


# =============================================================================
# CREACIÓN DE OBJETOS TOOL
# =============================================================================

everlast_rag_tool = Tool(
    name="BusquedaDocumentosEverlast",
    func=buscar_documentos_everlast,
    description=(
        "Busca información específica en la base de conocimiento interna de "
        "Everlast Chile. Contiene datos sobre: productos (guantes, vendas, sacos, "
        "ropa deportiva), especificaciones técnicas, tablas de tallas, políticas "
        "de envío, políticas de devolución, y recomendaciones de uso.\n\n"
        "**CUÁNDO USAR:** Usa esta herramienta SIEMPRE como primera opción para "
        "cualquier pregunta relacionada con Everlast, sus productos o servicios.\n\n"
        "**INPUT:** La pregunta específica del usuario en español.\n"
        "**OUTPUT:** Fragmentos de texto relevantes de los documentos internos."
    )
)

calculator_tool = Tool(
    name="CalculadoraSimple",
    func=simple_calculator,
    description=(
        "Calculadora para operaciones matemáticas básicas. Soporta suma (+), "
        "resta (-), multiplicación (*), división (/) y paréntesis.\n\n"
        "**CUÁNDO USAR:** Para calcular descuentos, conversiones de unidades, "
        "totales de compra, porcentajes, etc.\n\n"
        "**INPUT:** Expresión matemática como string. "
        "Ejemplos: '100 * 0.8', '(50+30)*1.19', '16 * 0.453592'\n"
        "**OUTPUT:** El resultado numérico del cálculo."
    )
)


def get_tools() -> List[Tool]:
    """
    Retorna la lista completa de herramientas disponibles para el agente.
    
    Returns:
        List[Tool]: Lista de herramientas configuradas
    """
    print("\n📦 Inicializando herramientas del agente...")
    
    # Verificar que el vector store existe
    try:
        obtener_vector_store()
        print("   ✅ Vector Store listo.")
    except Exception as e:
        print(f"   ⚠️  Advertencia: {e}")
    
    tools = [everlast_rag_tool, calculator_tool]
    print(f"   ✅ {len(tools)} herramientas disponibles: {[t.name for t in tools]}\n")
    
    return tools


# =============================================================================
# BLOQUE DE PRUEBAS STANDALONE
# =============================================================================

if __name__ == '__main__':
    """Pruebas para verificar el funcionamiento de las herramientas."""
    from dotenv import load_dotenv
    
    print("=" * 80)
    print("MODO DE PRUEBA - tools_everlast.py")
    print("=" * 80)
    
    load_dotenv()
    print("\n✅ Variables de entorno cargadas\n")
    
    # PRUEBA 1: RAG
    print("\n" + "=" * 80)
    print("PRUEBA 1: Búsqueda en Documentos")
    print("=" * 80)
    
    query_prueba = "¿Qué guantes recomiendas para sparring?"
    print(f"\n🔍 Query: '{query_prueba}'")
    
    try:
        resultado_rag = buscar_documentos_everlast(query_prueba)
        print("\n📄 RESULTADO:")
        print("-" * 80)
        print(resultado_rag[:500] + "..." if len(resultado_rag) > 500 else resultado_rag)
        print("-" * 80)
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # PRUEBA 2: Calculadora
    print("\n" + "=" * 80)
    print("PRUEBA 2: Calculadora")
    print("=" * 80)
    
    expresiones = [
        "150 * (1 - 0.15)",
        "(100 + 50) * 1.19",
        "16 * 0.453592"
    ]
    
    for expr in expresiones:
        resultado = simple_calculator(expr)
        print(f"   → {resultado}")
    
    # PRUEBA 3: get_tools()
    print("\n" + "=" * 80)
    print("PRUEBA 3: get_tools()")
    print("=" * 80)
    
    try:
        tools = get_tools()
        print(f"\n✅ {len(tools)} herramientas listas")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("FIN DE LAS PRUEBAS")
    print("=" * 80)