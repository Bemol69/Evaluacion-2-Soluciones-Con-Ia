"""
create_vectorstore.py
=====================
Script standalone para crear el índice vectorial FAISS desde los documentos .md
VERSIÓN ULTRA-COMPATIBLE - Soluciona problema de proxies

EJECUTAR UNA SOLA VEZ (o cuando cambien los documentos):
    python codigo/create_vectorstore.py

Autor: Evaluación 2 - Everlast Chile
"""

import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Importaciones de LangChain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Importar numpy para FAISS
import numpy as np

# Importar requests para llamadas HTTP directas
import requests
import json


def obtener_embeddings_http(textos: List[str], api_key: str, base_url: str) -> np.ndarray:
    """
    Genera embeddings usando requests HTTP directamente (sin cliente OpenAI).
    Esto evita problemas de incompatibilidad con proxies.
    
    Args:
        textos: Lista de textos para generar embeddings
        api_key: GitHub token
        base_url: URL base para embeddings
    
    Returns:
        np.ndarray: Matriz de embeddings
    """
    print(f"   Generando embeddings para {len(textos)} chunks...")
    
    # Construir URL completa
    url = f"{base_url.rstrip('/')}/embeddings"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(textos), batch_size):
            batch = textos[i:i+batch_size]
            print(f"   Procesando batch {i//batch_size + 1}/{(len(textos)-1)//batch_size + 1}...")
            
            # Payload
            payload = {
                "model": "text-embedding-3-small",
                "input": batch
            }
            
            # Hacer request HTTP POST
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Verificar respuesta
            if response.status_code != 200:
                raise Exception(
                    f"Error HTTP {response.status_code}: {response.text}"
                )
            
            # Parsear respuesta
            data = response.json()
            batch_embeddings = [item['embedding'] for item in data['data']]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error de conexión: {e}")
    except KeyError as e:
        raise Exception(f"Respuesta inválida de la API: {e}")
    except Exception as e:
        raise Exception(f"Error al generar embeddings: {e}")


def crear_faiss_index(embeddings: np.ndarray):
    """
    Crea un índice FAISS a partir de embeddings.
    
    Args:
        embeddings: Matriz de embeddings
    
    Returns:
        FAISS index
    """
    import faiss
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


def main():
    """Función principal para crear el vector store"""
    
    print("=" * 80)
    print("CREACIÓN DE VECTOR STORE FAISS - EVERLAST CHILE")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. CONFIGURACIÓN INICIAL
    # -------------------------------------------------------------------------
    print("\n[1/7] Cargando configuración...")
    
    load_dotenv()
    
    github_token = os.getenv("GITHUB_TOKEN")
    embeddings_url = os.getenv("OPENAI_EMBEDDINGS_URL")
    
    if not github_token:
        print("❌ ERROR: GITHUB_TOKEN no encontrado en .env")
        sys.exit(1)
    
    if not embeddings_url:
        print("❌ ERROR: OPENAI_EMBEDDINGS_URL no encontrado en .env")
        sys.exit(1)
    
    print(f"   ✅ Variables de entorno cargadas")
    print(f"   • GITHUB_TOKEN: ...{github_token[-8:]}")
    print(f"   • EMBEDDINGS_URL: {embeddings_url}")
    
    # -------------------------------------------------------------------------
    # 2. CONFIGURAR RUTAS
    # -------------------------------------------------------------------------
    print("\n[2/7] Configurando rutas...")
    
    if Path.cwd().name == "codigo":
        datos_folder = Path("../datos")
        vectorstore_path = Path("../datos/vectorstore_faiss")
    else:
        datos_folder = Path("datos")
        vectorstore_path = Path("datos/vectorstore_faiss")
    
    datos_folder = datos_folder.resolve()
    vectorstore_path = vectorstore_path.resolve()
    
    print(f"   ✅ Carpeta de datos: {datos_folder}")
    print(f"   ✅ Destino vector store: {vectorstore_path}")
    
    if not datos_folder.exists():
        print(f"\n❌ ERROR: La carpeta de datos no existe: {datos_folder}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 3. CARGAR DOCUMENTOS
    # -------------------------------------------------------------------------
    print("\n[3/7] Cargando documentos .md...")
    
    try:
        loader = DirectoryLoader(
            str(datos_folder),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            show_progress=True
        )
        
        documentos = loader.load()
        
        if not documentos:
            print(f"\n❌ ERROR: No se encontraron archivos .md en {datos_folder}")
            sys.exit(1)
        
        print(f"   ✅ {len(documentos)} documentos cargados")
        
        for i, doc in enumerate(documentos, 1):
            source = doc.metadata.get('source', 'unknown')
            filename = Path(source).name
            print(f"      {i}. {filename} ({len(doc.page_content)} caracteres)")
    
    except Exception as e:
        print(f"\n❌ ERROR al cargar documentos: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 4. DIVIDIR EN CHUNKS
    # -------------------------------------------------------------------------
    print("\n[4/7] Dividiendo documentos en chunks...")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Un tamaño máximo grande
    chunk_overlap=200,
    length_function=len,
    # Define los separadores en orden de prioridad
    separators=[
        "\n\n## ",  # Separar por títulos de Nivel 2 (SACOS, GUANTES, etc.)
        "\n\n### ", # Separar por títulos de Nivel 3 (Producto individual)
        "\n\n",     # Párrafos
        "\n",       # Líneas
        " "         # Palabras
    ]
)
        
        chunks = text_splitter.split_documents(documentos)
        
        print(f"   ✅ {len(chunks)} chunks creados")
        print(f"   • Tamaño de chunk: 1000 caracteres")
        print(f"   • Solapamiento: 200 caracteres")
    
    except Exception as e:
        print(f"\n❌ ERROR al dividir documentos: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 5. GENERAR EMBEDDINGS
    # -------------------------------------------------------------------------
    print("\n[5/7] Generando embeddings con HTTP directo...")
    print("   ⏳ Esto puede tomar 30-60 segundos...")
    print("   💡 Usando requests HTTP (evita problemas de proxies)")
    
    try:
        # Extraer textos de los chunks
        textos = [chunk.page_content for chunk in chunks]
        
        # Generar embeddings usando HTTP directo
        embeddings_matrix = obtener_embeddings_http(textos, github_token, embeddings_url)
        
        print(f"   ✅ Embeddings generados: {embeddings_matrix.shape}")
    
    except Exception as e:
        print(f"\n❌ ERROR al generar embeddings: {e}")
        print("\nDetalles del error:")
        print(f"   {str(e)}")
        print("\nPosibles causas:")
        print("  • Token de GitHub inválido o expirado")
        print("  • Problema de conexión a internet")
        print("  • URL de embeddings incorrecta")
        print("\n💡 Verifica tu token en: https://github.com/settings/tokens")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 6. CREAR ÍNDICE FAISS
    # -------------------------------------------------------------------------
    print("\n[6/7] Creando índice FAISS...")
    
    try:
        faiss_index = crear_faiss_index(embeddings_matrix)
        print(f"   ✅ Índice FAISS creado ({faiss_index.ntotal} vectores)")
    
    except Exception as e:
        print(f"\n❌ ERROR al crear índice FAISS: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 7. GUARDAR EN DISCO
    # -------------------------------------------------------------------------
    print("\n[7/7] Guardando en disco...")
    
    try:
        # Crear carpeta si no existe
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        # Verificar que la carpeta existe y tiene permisos
        if not vectorstore_path.exists():
            raise Exception(f"No se pudo crear la carpeta: {vectorstore_path}")
        
        print(f"   📁 Carpeta verificada: {vectorstore_path}")
        
        # Guardar índice FAISS
        import faiss
        faiss_file = str(vectorstore_path / "index.faiss")
        print(f"   💾 Guardando índice FAISS...")
        
        # Intentar con diferentes métodos según el OS
        try:
            faiss.write_index(faiss_index, faiss_file)
        except Exception as e1:
            print(f"      ⚠️ Primer intento falló, probando método alternativo...")
            # Intentar guardando en un archivo temporal primero
            import tempfile
            import shutil
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.faiss') as tmp:
                tmp_path = tmp.name
            faiss.write_index(faiss_index, tmp_path)
            shutil.move(tmp_path, faiss_file)
        
        print(f"   ✅ Índice FAISS guardado")
        
        # Guardar chunks (metadatos)
        with open(vectorstore_path / "chunks.pkl", 'wb') as f:
            pickle.dump(chunks, f)
        
        # Guardar configuración
        config = {
            'model': 'text-embedding-3-small',
            'dimension': embeddings_matrix.shape[1],
            'num_chunks': len(chunks)
        }
        with open(vectorstore_path / "config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"   ✅ Vector store guardado en: {vectorstore_path}")
        
        # Verificar archivos creados
        files = list(vectorstore_path.glob("*"))
        print(f"   • Archivos creados: {len(files)}")
        for file in files:
            size_kb = file.stat().st_size / 1024
            print(f"      - {file.name} ({size_kb:.1f} KB)")
    
    except Exception as e:
        print(f"\n❌ ERROR al guardar: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # FINALIZACIÓN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ VECTOR STORE CREADO EXITOSAMENTE")
    print("=" * 80)
    print("\nPróximos pasos:")
    print("  1. Probar herramientas: python codigo/tools_everlast.py")
    print("  2. Ejecutar agente: python codigo/agente_principal.py")
    print("  3. O ejecutar Streamlit: streamlit run codigo/app_streamlit.py")
    print("\n💡 Este script solo necesita ejecutarse una vez")
    print("   (o cuando actualices los archivos .md)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)