# 🥊 AGENTE INTELIGENTE EVERLAST CHILE
**Evaluación Parcial 2 - Desarrollo de Agente Funcional**

---

## 📋 INFORMACIÓN DEL PROYECTO

- **Asignatura**: Inteligencia Artificial
- **Módulo**: IL2 - Sistemas de Agentes Inteligentes
- **Indicadores de Logro**: IL2.1, IL2.2, IL2.3, IL2.4
- **Estudiantes**: Bryan Piña y Juan Castro
- **Fecha**: Octubre 2025
- **Repositorio**: 

---

## 🎯 OBJETIVO DEL PROYECTO

Desarrollar un **agente conversacional inteligente** para Everlast Chile que automatice tareas cognitivas complejas en el proceso de atención al cliente, específicamente:

1. **Consulta de productos**: Búsqueda semántica en catálogo de equipamiento deportivo
2. **Asesoría técnica**: Recomendaciones de tallas, especificaciones y uso adecuado
3. **Cálculos comerciales**: Descuentos, conversiones de unidades, totales de compra
4. **Gestión de políticas**: Información sobre envíos, devoluciones y garantías

---

## 🏗️ ARQUITECTURA DEL SISTEMA

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIO FINAL                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   INTERFAZ DE INTERACCIÓN     │
         │  • CLI (agente_principal.py)  │
         │  • Web (app_streamlit.py)     │
         └───────────────┬───────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    AGENTE PRINCIPAL                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              CUSTOM LLM (GPT-4o-mini)                    │  │
│  │  • Modelo: gpt-4o-mini via GitHub Models                 │  │
│  │  • Comunicación: HTTP directo (requests)                 │  │
│  │  • Prompt Engineering: Sistema + Instrucciones           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           SISTEMA DE MEMORIA (IL2.2)                     │  │
│  │  • Short-term: MemoriaSimple (buffer conversacional)     │  │
│  │  • Contexto: Últimos 10 mensajes                         │  │
│  │  • Persistencia: Historial en sesión activa              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      MOTOR DE DECISIÓN Y PLANIFICACIÓN (IL2.3)           │  │
│  │  • Análisis de consulta del usuario                      │  │
│  │  • Selección de herramienta apropiada                    │  │
│  │  • Ejecución iterativa (máx 3 pasos)                     │  │
│  │  • Validación y formateo de respuesta                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                  CAPA DE HERRAMIENTAS (IL2.1)                   │
│                                                                  │
│  ┌──────────────────────────┐  ┌────────────────────────────┐  │
│  │ BusquedaDocumentosEverlast│  │    CalculadoraSimple      │  │
│  │ • RAG con FAISS           │  │ • Operaciones básicas     │  │
│  │ • Embeddings: text-emb-3  │  │ • Validación de input     │  │
│  │ • Top-K: 3 chunks         │  │ • Manejo de errores       │  │
│  └────────────┬─────────────┘  └────────────────────────────┘  │
└───────────────┼────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────┐
│              BASE DE CONOCIMIENTO VECTORIAL                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Vector Store FAISS (datos/vectorstore_faiss/)           │  │
│  │  • index.faiss: Índice de 9 vectores (1536 dims)         │  │
│  │  • chunks.pkl: Fragmentos de documentos                  │  │
│  │  • config.pkl: Metadatos del modelo                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Documentos Fuente (datos/*.md)                          │  │
│  │  • productos.md: Catálogo completo                       │  │
│  │  • tallas.md: Guía de tallas por peso                    │  │
│  │  • politicas.md: Envíos, devoluciones, garantías        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔧 COMPONENTES PRINCIPALES

### 1. **Agente Principal** (`agente_principal.py`)

**Responsabilidad**: Orquestación del flujo conversacional y toma de decisiones.

**Características IL2.3 - Planificación**:
- **Análisis de intención**: Interpreta si la consulta requiere herramientas o respuesta directa
- **Planificación multi-paso**: Ejecuta hasta 3 iteraciones de razonamiento-acción-observación
- **Adaptación dinámica**: Ajusta estrategia según resultados intermedios
- **Manejo de errores**: Recuperación ante fallos de herramientas o API

**Flujo de Decisión**:
```
Usuario → LLM (análisis) → ¿Necesita herramienta?
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   SÍ                  NO
                    │                   │
                    ▼                   ▼
           Ejecutar herramienta    Responder
                    │               directamente
                    ▼
           Procesar resultado
                    │
                    ▼
           LLM (respuesta final)
```

**Código clave**:
```python
class AgenteEverlast:
    def procesar(self, consulta_usuario: str) -> str:
        # 1. Agregar a memoria
        self.memoria.agregar("user", consulta_usuario)
        
        # 2. Preparar contexto con historial
        mensajes = [{"role": "system", "content": self.system_prompt}]
        mensajes.extend(self.memoria.obtener_historial())
        
        # 3. Ciclo de razonamiento (máx 3 iteraciones)
        for iteracion in range(3):
            respuesta_llm = self._llamar_llm(mensajes)
            
            # 4. Detectar si necesita herramienta
            if "USAR_HERRAMIENTA:" in respuesta_llm:
                resultado = self._ejecutar_herramienta(...)
                mensajes.append(resultado)
            else:
                break  # Respuesta lista
        
        return respuesta_final
```

---

### 2. **Sistema de Memoria** (IL2.2)

**Implementación**: `MemoriaSimple` en `agente_principal.py`

**Tipos de Memoria**:

#### 📝 **Short-Term Memory**
- **Propósito**: Mantener contexto de conversación activa
- **Implementación**: Buffer circular de últimos 10 mensajes
- **Persistencia**: Durante sesión activa
- **Uso**: Responder preguntas de seguimiento, mantener coherencia

```python
class MemoriaSimple:
    def __init__(self):
        self.historial = []
    
    def agregar(self, rol: str, contenido: str):
        self.historial.append({"role": rol, "content": contenido})
    
    def obtener_historial(self, ultimos_n: int = 10):
        return self.historial[-ultimos_n:]  # Ventana deslizante
```

**Ejemplo de uso**:
```
Usuario: "¿Qué guantes recomiendas?"
Agente: "Para principiantes recomiendo los Pro Style..."

Usuario: "¿Y en qué tallas vienen?" ← Memoria permite entender "esos" guantes
Agente: "Los Pro Style vienen en 12oz, 14oz y 16oz..."
```

#### 🗄️ **Long-Term Memory** (Implementación mediante Vector Store)
- **Propósito**: Conocimiento persistente sobre productos/políticas
- **Implementación**: FAISS con embeddings
- **Recuperación**: Búsqueda semántica (RAG)
- **Actualización**: Re-indexación al modificar documentos fuente

---

### 3. **Herramientas** (`tools_everlast.py`) - IL2.1

#### 🔍 **BusquedaDocumentosEverlast**

**Tecnología**: Retrieval-Augmented Generation (RAG)

**Pipeline**:
```
Query → Embedding (text-embedding-3-small) → FAISS search → Top-3 chunks → Contexto
```

**Especificaciones técnicas**:
- **Vector Store**: FAISS IndexFlatL2 (búsqueda L2)
- **Dimensión**: 1536 (text-embedding-3-small)
- **Chunk size**: 1000 caracteres
- **Overlap**: 200 caracteres
- **Top-K**: 3 resultados más relevantes

**Código**:
```python
def buscar_similares(query: str, k: int = 3) -> List[str]:
    # 1. Generar embedding de query
    query_embedding = obtener_embedding_http(query)
    
    # 2. Búsqueda en FAISS
    distances, indices = index.search(query_embedding, k)
    
    # 3. Recuperar chunks
    resultados = [chunks[idx].page_content for idx in indices[0]]
    
    return resultados
```

#### 🔢 **CalculadoraSimple**

**Capacidades**:
- Operaciones aritméticas: `+, -, *, /`
- Paréntesis para precedencia
- Números decimales
- Validación de seguridad (solo operadores permitidos)

**Casos de uso**:
- Cálculo de descuentos: `150 * (1 - 0.15)` → precio con 15% off
- Conversión de libras a kg: `16 * 0.453592`
- Totales con IVA: `(100 + 50) * 1.19`

---

### 4. **Generación de Vector Store** (`create_vectorstore.py`)

**Proceso de creación** (ejecutar una sola vez):

```bash
python codigo/create_vectorstore.py
```

**Pasos internos**:
1. **Carga de documentos**: DirectoryLoader para archivos `.md`
2. **Chunking**: RecursiveCharacterTextSplitter (1000/200)
3. **Embeddings**: HTTP POST a GitHub Models API
4. **Indexación**: FAISS IndexFlatL2
5. **Serialización**: Pickle para chunks y config

**Archivos generados**:
- `datos/vectorstore_faiss/index.faiss`: Índice vectorial (54 KB)
- `datos/vectorstore_faiss/chunks.pkl`: Metadatos de chunks (6.8 KB)
- `datos/vectorstore_faiss/config.pkl`: Configuración del modelo (0.1 KB)

---

## 🚀 INSTALACIÓN Y USO

### Requisitos Previos

- Python 3.9+
- Git
- Cuenta GitHub (para GitHub Models API)

### Instalación

```bash
# 1. Clonar repositorio
git clone [URL_REPOSITORIO]
cd "Evaluación 2"

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno (Windows)
.\venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Configurar variables de entorno
# Crear archivo .env con:
GITHUB_TOKEN=tu_token_aqui
OPENAI_BASE_URL=https://models.inference.ai.azure.com
OPENAI_EMBEDDINGS_URL=https://models.github.ai/inference
```

### Configuración Inicial

```bash
# Crear vector store (SOLO UNA VEZ)
python codigo/create_vectorstore.py
```

**Salida esperada**:
```
✅ 3 documentos cargados
✅ 9 chunks creados
✅ Embeddings generados: (9, 1536)
✅ Índice FAISS creado (9 vectores)
✅ Vector store guardado
```

### Ejecución

#### Opción 1: Interfaz de Consola

```bash
python codigo/agente_principal.py
```

**Comandos disponibles**:
- `salir` / `exit`: Terminar sesión
- `limpiar` / `clear`: Borrar memoria
- `historial`: Ver conversación completa

#### Opción 2: Interfaz Web (Streamlit)

```bash
streamlit run codigo/app_streamlit.py
```

Abre navegador en `http://localhost:8501`

---

## 📊 ANÁLISIS DEL FLUJO DE TRABAJO

### Proceso Organizacional: Atención al Cliente Everlast

**Tareas Cognitivas Identificadas**:

1. **Búsqueda de información de productos** (Alta complejidad)
   - **Problema**: Catálogo de varios productos con especificaciones técnicas
   - **Solución**: RAG con búsqueda semántica en documentos internos
   - **Automatización**: Respuestas instantáneas con contexto relevante

2. **Asesoría personalizada en tallas** (Complejidad media)
   - **Problema**: Tabla de tallas variable según peso y tipo de producto
   - **Solución**: Recuperación de guía de tallas + razonamiento del LLM
   - **Automatización**: Recomendaciones basadas en perfil del usuario

3. **Cálculos comerciales** (Baja complejidad)
   - **Problema**: Descuentos, conversiones, totales con impuestos
   - **Solución**: Herramienta de calculadora integrada
   - **Automatización**: Cálculos precisos sin intervención humana

4. **Consultas de políticas** (Alta repetición)
   - **Problema**: Mismas preguntas sobre envíos/devoluciones
   - **Solución**: Base de conocimiento vectorial
   - **Automatización**: 100% de consultas de política automatizables

---

## 🎓 CUMPLIMIENTO DE INDICADORES DE LOGRO

### ✅ IL2.1: Construcción de Agente Funcional

**Framework utilizado**: LangChain + Custom LLM

**Herramientas implementadas**:
1. ✅ **Consulta**: `BusquedaDocumentosEverlast` (RAG)
2. ✅ **Razonamiento**: LLM GPT-4o-mini para interpretación
3. ✅ **Cálculo**: `CalculadoraSimple` para operaciones

**Evidencia**:
- Archivo `tools_everlast.py`:  (definición de herramientas)
- Archivo `agente_principal.py`:  (integración de herramientas)

---

### ✅ IL2.2: Sistema de Memoria

**Short-Term Memory**:
- **Implementación**: `MemoriaSimple` ( `agente_principal.py`)
- **Capacidad**: 10 mensajes recientes
- **Función**: Mantener contexto conversacional

**Long-Term Memory**:
- **Implementación**: Vector Store FAISS
- **Persistencia**: Disco (carpeta `datos/vectorstore_faiss/`)
- **Función**: Conocimiento persistente de productos/políticas

**Recuperación de contexto**:
- Método `obtener_historial()`: Recupera ventana de conversación
- Método `buscar_similares()`: Recupera conocimiento relevante

**Evidencia**:
- Short-term: `agente_principal.py`, clase `MemoriaSimple`
- Long-term: `create_vectorstore.py`, generación de índice FAISS

---

### ✅ IL2.3: Planificación y Toma de Decisiones

**Estrategia implementada**: ReAct simplificado (Reasoning + Acting)

**Componentes**:

1. **Análisis de intención** :
   ```python
   mensajes = [{"role": "system", "content": self.system_prompt}]
   mensajes.extend(self.memoria.obtener_historial())
   ```

2. **Ciclo de razonamiento iterativo** (líneas 107-135):
   - **Iteración 1**: Determina si necesita herramienta
   - **Iteración 2-3**: Ejecuta herramienta y procesa resultado
   - **Máximo**: 3 iteraciones para evitar loops infinitos

3. **Ajuste dinámico**:
   - Si "USAR_HERRAMIENTA" detectado → ejecuta y reintenta
   - Si "RESPUESTA" directa → finaliza y responde
   - Si error → manejo de excepciones y mensaje alternativo

**Condiciones cambiantes**:
- Consultas simples (saludo) → Respuesta directa sin herramientas
- Consultas de productos → Activación de RAG
- Consultas con cálculos → Activación de calculadora
- Consultas complejas → Combinación de herramientas

**Evidencia**: `agente_principal.py`, método `procesar()` (líneas 89-145)

---

### ✅ IL2.4: Documentación Técnica

**Este README.md cumple con**:
- ✅ Descripción completa del diseño
- ✅ Diagrama de arquitectura de componentes
- ✅ Explicación de orquestación entre módulos
- ✅ Relación con flujo automatizado (atención al cliente)
- ✅ Instrucciones de instalación y uso
- ✅ Evidencia de código por indicador de logro

---

## 📁 ESTRUCTURA DEL PROYECTO

```
Evaluación 2/
├── codigo/
│   ├── agente_principal.py       # Agente principal con memoria y planificación
│   ├── app_streamlit.py          # Interfaz web interactiva
│   ├── create_vectorstore.py     # Script de creación de índice FAISS
│   └── tools_everlast.py         # Definición de herramientas (RAG + Calculadora)
│
├── datos/
│   ├── productos.md              # Catálogo de productos
│   ├── tallas.md                 # Guía de tallas
│   ├── politicas.md              # Políticas de envío y devolución
│   └── vectorstore_faiss/        # Índice vectorial generado
│       ├── index.faiss
│       ├── chunks.pkl
│       └── config.pkl
│
├── documentacion/
│   └── README (archivos de referencia)
│
├── .env                          # Variables de entorno (NO subir a Git)
├── .gitignore                    # Exclusiones de Git
├── requirements.txt              # Dependencias del proyecto
├── Pasos.txt                     # Guía rápida de ejecución
└── README.md                     # Este archivo
```

---

## 🧪 PRUEBAS Y VALIDACIÓN

### Casos de Prueba

#### Test 1: Consulta de producto
```
Usuario: "¿Qué guantes recomiendas para sparring?"
Esperado: Activación de BusquedaDocumentosEverlast + recomendación de 14oz o 16oz
```

#### Test 2: Memoria conversacional
```
Usuario: "Hola"
Agente: "¡Hola! ¿Cómo puedo ayudarte?"
Usuario: "¿Cuánto cuesta un saco de boxeo?"
Agente: [Busca en documentos]
Usuario: "¿Y viene con cadena?"  ← Requiere recordar "saco de boxeo"
Esperado: Respuesta contextual sin re-preguntar qué producto
```

#### Test 3: Cálculo de descuento
```
Usuario: "Si un guante cuesta $45.990 y tiene 20% de descuento, ¿cuánto pagaría?"
Esperado: Uso de CalculadoraSimple → "45990 * (1 - 0.20)" = $36.792
```

#### Test 4: Comando de limpieza
```
Usuario: "limpiar"
Esperado: Memoria borrada, siguiente consulta no tiene contexto previo
```

### Ejecución de Tests

```bash
# Probar herramientas standalone
python codigo/tools_everlast.py

# Salida esperada:
# ✅ 3 fragmentos recuperados (RAG)
# ✅ Calculadora: 127.5, 178.5, 7.257472
```

---

## 🛠️ TECNOLOGÍAS UTILIZADAS

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| Lenguaje | Python | 3.9+ |
| Framework Agentes | LangChain | 0.1.20 |
| LLM | GPT-4o-mini | GitHub Models |
| Embeddings | text-embedding-3-small | GitHub Models |
| Vector Store | FAISS | 1.8.0 |
| Web Framework | Streamlit | 1.32.0 |
| HTTP Client | Requests | 2.31.0 |
| Gestión de Env | python-dotenv | 1.0.1 |

---

## ⚠️ LIMITACIONES CONOCIDAS

1. **Memoria long-term**: No hay persistencia entre sesiones (se pierde al cerrar)
2. **Escalabilidad**: FAISS IndexFlatL2 no es óptimo para >100K vectores
3. **Multilenguaje**: Solo español, sin soporte para otros idiomas
4. **Validación de entrada**: Calculadora no detecta expresiones maliciosas complejas

---

## 🚀 MEJORAS FUTURAS

1. **Memoria persistente**: Integrar SQLite o Redis para historial multi-sesión
2. **Planificación avanzada**: Implementar algoritmo MCTS o A* para tareas complejas
3. **Herramientas adicionales**: 
   - Consulta de stock en tiempo real (API)
   - Generación de cotizaciones en PDF
   - Integración con sistema de tickets
4. **Evaluación de calidad**: Métricas de satisfacción del usuario

---

## 👥 AUTORES

- **Bryan Piña** - Desarrollo de agente y herramientas
- **Juan Castro** - Sistema de memoria y vector store

---

## 📄 LICENCIA

Este proyecto es parte de una evaluación académica.  
Institución: Duoc Uc  
Asignatura: Doluciones con Inteligencia Artificial 

---

**Versión**: 1.0.0