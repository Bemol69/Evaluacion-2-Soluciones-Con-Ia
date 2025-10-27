"""
app_streamlit.py
================
Interfaz web SIMPLIFICADA para el agente Everlast.
Compatible con todas las versiones.

Ejecutar: streamlit run codigo/app_streamlit.py

Autor: Evaluación 2 - Everlast Chile
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Agregar path del código
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agente_principal import AgenteEverlast


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Asistente Everlast",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# ESTILOS CSS (CORREGIDOS)
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E31837;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Se eliminó la regla .stChatMessage que causaba el fondo blanco.
    Streamlit ahora usará el estilo de tema oscuro por defecto.
    */

    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INICIALIZACIÓN
# =============================================================================

def inicializar_agente():
    """Inicializa el agente una sola vez"""
    if "agente" not in st.session_state:
        load_dotenv()
        
        github_token = os.getenv("GITHUB_TOKEN")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not github_token or not base_url:
            st.error("❌ Error: Variables de entorno no configuradas")
            st.stop()
        
        st.session_state.agente = AgenteEverlast(github_token, base_url)
    
    # Inicializar historial de mensajes si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []


# =============================================================================
# INTERFAZ PRINCIPAL (LÓGICA CORREGIDA)
# =============================================================================

def main():
    # Inicializar
    inicializar_agente()
    
    # Header
    st.markdown('<div class="main-header">🥊 ASISTENTE EVERLAST CHILE</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        
        # Información
        st.info("""
        **Asistente inteligente de Everlast**
        
        Te ayudo con:
        - 🔍 Información de productos
        - 📏 Guía de tallas
        - 🔢 Cálculos de precios
        - 📦 Políticas de envío
        """)
        
        st.markdown("---")
        
        # Botón para limpiar
        if st.button("🧹 Nueva Conversación", use_container_width=True):
            st.session_state.agente.memoria.limpiar()
            st.session_state.messages = []
            st.success("✅ Memoria limpiada")
            st.rerun()
        
        # Estadísticas
        st.markdown("---")
        st.subheader("📊 Estadísticas")
        st.metric("Mensajes", len(st.session_state.messages))
        
        # Ejemplos
        st.markdown("---")
        st.subheader("💡 Prueba preguntar:")
        
        ejemplos = [
            "¿Qué guantes recomiendas para principiantes?",
            "¿Cuál es mi talla si peso 75kg?",
            "Calcula 15% de descuento en $50.000",
            "¿Cuál es la política de devolución?"
        ]
        
        for ejemplo in ejemplos:
            if st.button(f"📝 {ejemplo[:30]}...", key=ejemplo, use_container_width=True):
                # CORRECCIÓN: Solo agregar el mensaje y re-ejecutar
                st.session_state.messages.append({
                    "role": "user",
                    "content": ejemplo
                })
                st.rerun()
    
    # Área principal de chat
    
    # 1. Mostrar historial de mensajes
    # Esto mostrará todos los mensajes, incluidos los de los botones
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Lógica de procesamiento (SEPARADA)
    # Si el último mensaje es del usuario, el agente debe responder.
    # Esto funciona tanto para los botones como para el chat_input
    
    try:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            # Obtener la consulta del usuario
            user_query = st.session_state.messages[-1]["content"]
            
             # Mostrar spinner y procesar
            with st.chat_message("assistant"):
                with st.spinner("🤔 Pensando..."):
                    try:
                        respuesta = st.session_state.agente.procesar(user_query)
                        st.markdown(respuesta)
                    except Exception as e:
                        respuesta = f"❌ Error al procesar: {str(e)[:200]}"
                        st.error(respuesta)
                    
                    
    except Exception as e:
        error_msg = f"❌ Error al procesar: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg
        })

    # 3. Input del usuario (al final)
    if prompt := st.chat_input("Escribe tu pregunta sobre Everlast..."):
        # CORRECCIÓN: Solo agregar el mensaje y re-ejecutar
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.rerun()


if __name__ == "__main__":
    main()