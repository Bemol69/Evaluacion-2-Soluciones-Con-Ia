"""
agente_principal.py
===================
Agente conversacional con ciclo ReAct MEJORADO Y ROBUSTO
Manejo avanzado de errores y parsing flexible

Ejecutar: python codigo/agente_principal.py

Autor: Evaluación 2 - Everlast Chile
Versión: 2.0 (Sin errores 400)
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional

from tools_everlast import buscar_documentos_everlast, simple_calculator


# =============================================================================
# MEMORIA SIMPLE
# =============================================================================

class MemoriaSimple:
    """Memoria conversacional básica con límite de caracteres"""
    def __init__(self, max_mensajes: int = 10, max_chars_por_mensaje: int = 1000):
        self.historial = []
        self.max_mensajes = max_mensajes
        self.max_chars = max_chars_por_mensaje
    
    def agregar(self, rol: str, contenido: str):
        """Agrega mensaje truncando si es muy largo"""
        if len(contenido) > self.max_chars:
            contenido = contenido[:self.max_chars] + "... [truncado]"
        
        self.historial.append({"role": rol, "content": contenido})
    
    def obtener_historial(self, ultimos_n: Optional[int] = None):
        """Obtiene últimos N mensajes"""
        n = ultimos_n or self.max_mensajes
        return self.historial[-n:]
    
    def limpiar(self):
        """Limpia toda la memoria"""
        self.historial = []
    
    def contar_tokens_aprox(self) -> int:
        """Estimación aproximada de tokens (4 chars = 1 token)"""
        total_chars = sum(len(msg["content"]) for msg in self.historial)
        return total_chars // 4


# =============================================================================
# AGENTE PRINCIPAL CON MANEJO ROBUSTO DE ERRORES
# =============================================================================

class AgenteEverlast:
    """Agente conversacional para Everlast Chile con manejo avanzado de errores"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.memoria = MemoriaSimple(max_mensajes=8, max_chars_por_mensaje=800)
        
        # Prompt mejorado y más conciso
        self.system_prompt = """Eres un asistente experto de Everlast Chile.

HERRAMIENTAS:
1. buscar_documentos - Busca en catálogo de productos/políticas
2. calcular - Hace cálculos matemáticos

FORMATO DE RESPUESTA:

Si necesitas herramienta:
HERRAMIENTA: nombre_herramienta
INPUT: texto_del_input

Si ya puedes responder:
RESPUESTA: tu respuesta aquí

REGLAS:
- Para preguntas de productos/tallas/políticas → USA buscar_documentos
- Para cálculos (descuentos, totales) → USA calcular
- Para saludos/charla → RESPUESTA directa
- Sé breve y directo

Ejemplos:

Usuario: "¿Qué guantes recomiendas?"
HERRAMIENTA: buscar_documentos
INPUT: guantes recomendados principiantes

Usuario: "Calcula 20% descuento de $50000"
HERRAMIENTA: calcular
INPUT: 50000 * 0.8

Usuario: "Hola"
RESPUESTA: ¡Hola! Soy tu asistente de Everlast. ¿En qué puedo ayudarte?"""
    
    def _llamar_llm(self, mensajes: List[Dict], reintentos: int = 2) -> str:
        """
        Llama al LLM vía HTTP con manejo robusto de errores.
        
        Args:
            mensajes: Lista de mensajes del chat
            reintentos: Número de reintentos en caso de error
        
        Returns:
            Respuesta del LLM o mensaje de error amigable
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Limpiar y validar mensajes
        mensajes_limpios = []
        for msg in mensajes:
            if msg.get("content") and msg["content"].strip():
                # Truncar mensajes muy largos
                content = msg["content"]
                if len(content) > 1500:
                    content = content[:1500] + "... [truncado]"
                
                mensajes_limpios.append({
                    "role": msg["role"],
                    "content": content
                })
        
        # Limitar total de mensajes para evitar exceder límite de tokens
        if len(mensajes_limpios) > 12:
            # Mantener system + últimos 10 mensajes
            mensajes_limpios = [mensajes_limpios[0]] + mensajes_limpios[-10:]
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": mensajes_limpios,
            "temperature": 0.5,
            "max_tokens": 800,  # Limitar respuesta para evitar timeouts
            "top_p": 0.9
        }
        
        # Intentar llamada con reintentos
        for intento in range(reintentos + 1):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=30  # Timeout de 30 segundos
                )
                
                # Manejar códigos de error HTTP
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                
                elif response.status_code == 400:
                    print(f"⚠️  Error 400 (Bad Request): {response.text[:200]}")
                    return "Disculpa, tu consulta es muy compleja. Intenta simplificarla."
                
                elif response.status_code == 401:
                    return "❌ Error de autenticación. Token inválido o expirado."
                
                elif response.status_code == 429:
                    print(f"⚠️  Rate limit alcanzado. Esperando {intento + 1} segundos...")
                    time.sleep(intento + 1)
                    continue
                
                elif response.status_code >= 500:
                    print(f"⚠️  Error del servidor ({response.status_code}). Reintentando...")
                    time.sleep(1)
                    continue
                
                else:
                    return f"Error técnico (código {response.status_code}). Por favor intenta de nuevo."
            
            except requests.exceptions.Timeout:
                print(f"⚠️  Timeout en intento {intento + 1}. Reintentando...")
                if intento < reintentos:
                    time.sleep(1)
                    continue
                return "La consulta tardó demasiado. Intenta con una pregunta más simple."
            
            except requests.exceptions.ConnectionError:
                return "❌ Error de conexión. Verifica tu internet."
            
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                return f"Ocurrió un error al procesar tu consulta. Intenta reformularla."
        
        return "No pude procesar tu consulta después de varios intentos."
    
    def _parsear_respuesta_llm(self, respuesta: str) -> Dict:
        """
        Parsea la respuesta del LLM de forma flexible.
        
        Returns:
            Dict con 'tipo': 'herramienta' o 'respuesta' y datos asociados
        """
        respuesta_upper = respuesta.upper()
        
        # Detectar uso de herramienta (múltiples variaciones)
        if any(keyword in respuesta_upper for keyword in [
            "HERRAMIENTA:", "USAR_HERRAMIENTA:", "TOOL:", "ACTION:"
        ]):
            try:
                lineas = respuesta.split('\n')
                nombre_herramienta = None
                input_herramienta = None
                
                for i, linea in enumerate(lineas):
                    linea_upper = linea.strip().upper()
                    
                    # Detectar nombre de herramienta
                    if any(k in linea_upper for k in ["HERRAMIENTA:", "TOOL:", "ACTION:"]):
                        # Extraer valor después de ':'
                        partes = linea.split(':', 1)
                        if len(partes) > 1:
                            nombre_herramienta = partes[1].strip().lower()
                            # Normalizar nombres
                            if "buscar" in nombre_herramienta or "documento" in nombre_herramienta:
                                nombre_herramienta = "buscar_documentos"
                            elif "calcul" in nombre_herramienta:
                                nombre_herramienta = "calcular"
                    
                    # Detectar input
                    if "INPUT:" in linea_upper or "QUERY:" in linea_upper:
                        partes = linea.split(':', 1)
                        if len(partes) > 1:
                            input_herramienta = partes[1].strip()
                        # Si el input viene en la siguiente línea
                        elif i + 1 < len(lineas):
                            input_herramienta = lineas[i + 1].strip()
                
                if nombre_herramienta and input_herramienta:
                    return {
                        'tipo': 'herramienta',
                        'nombre': nombre_herramienta,
                        'input': input_herramienta
                    }
            
            except Exception as e:
                print(f"⚠️  Error al parsear herramienta: {e}")
        
        # Detectar respuesta directa
        if "RESPUESTA:" in respuesta_upper:
            try:
                respuesta_texto = respuesta.split("RESPUESTA:", 1)[1].strip()
                return {
                    'tipo': 'respuesta',
                    'contenido': respuesta_texto
                }
            except:
                pass
        
        # Fallback: asumir que toda la respuesta es la respuesta final
        return {
            'tipo': 'respuesta',
            'contenido': respuesta
        }
    
    def _ejecutar_herramienta(self, nombre: str, input_str: str) -> str:
        """Ejecuta una herramienta con manejo de errores"""
        print(f"🔧 Ejecutando: {nombre}")
        print(f"   Input: {input_str[:100]}...")
        
        try:
            if nombre == "buscar_documentos":
                resultado = buscar_documentos_everlast(input_str)
            elif nombre == "calcular":
                resultado = simple_calculator(input_str)
            else:
                resultado = f"⚠️ Herramienta desconocida: {nombre}"
            
            # Truncar resultados muy largos
            if len(resultado) > 2000:
                resultado = resultado[:2000] + "\n... [resultado truncado]"
            
            print(f"   ✅ Resultado: {len(resultado)} caracteres")
            return resultado
            
        except Exception as e:
            error_msg = f"Error al ejecutar {nombre}: {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg
    
    def procesar(self, consulta_usuario: str) -> str:
        """
        Procesa consulta con ciclo ReAct mejorado y robusto.
        
        Args:
            consulta_usuario: Pregunta del usuario
        
        Returns:
            Respuesta final del agente
        """
        print("\n" + "="*80)
        print(f"🤖 PROCESANDO: {consulta_usuario[:100]}...")
        print("="*80)
        
        # Validar input
        if not consulta_usuario or len(consulta_usuario.strip()) == 0:
            return "Por favor, escribe una pregunta válida."
        
        # Truncar consultas muy largas
        if len(consulta_usuario) > 500:
            consulta_usuario = consulta_usuario[:500] + "..."
        
        # 1. Agregar a memoria
        self.memoria.agregar("user", consulta_usuario)
        
        # 2. Preparar mensajes
        mensajes = [{"role": "system", "content": self.system_prompt}]
        
        # Agregar solo últimos 6 mensajes de historial
        historial = self.memoria.obtener_historial(6)
        for msg in historial[:-1]:  # Excluir consulta actual
            mensajes.append(msg)
        
        # Agregar consulta actual
        mensajes.append({"role": "user", "content": consulta_usuario})
        
        # 3. Ciclo ReAct (máximo 2 iteraciones)
        respuesta_final = None
        
        for iteracion in range(1, 3):  # Solo 2 iteraciones
            print(f"\n🔄 Iteración {iteracion}/2")
            
            # Llamar al LLM
            respuesta_llm = self._llamar_llm(mensajes)
            
            # Si hubo error en la llamada, retornar error
            if respuesta_llm.startswith("❌") or respuesta_llm.startswith("Error"):
                return respuesta_llm
            
            print(f"💬 LLM responde: {respuesta_llm[:150]}...")
            
            # Parsear respuesta
            parsed = self._parsear_respuesta_llm(respuesta_llm)
            
            if parsed['tipo'] == 'herramienta':
                # Ejecutar herramienta
                resultado_tool = self._ejecutar_herramienta(
                    parsed['nombre'],
                    parsed['input']
                )
                
                # Agregar resultado al contexto
                observacion = f"Resultado de {parsed['nombre']}:\n{resultado_tool}"
                mensajes.append({"role": "assistant", "content": respuesta_llm})
                mensajes.append({"role": "user", "content": observacion})
                
                print(f"   ✅ Herramienta ejecutada")
                continue  # Siguiente iteración
            
            else:
                # Respuesta directa
                respuesta_final = parsed['contenido']
                print(f"✅ Respuesta final lista")
                break
        
        else:
            # Se alcanzó límite de iteraciones
            print("⚠️  Límite de iteraciones alcanzado")
            if respuesta_llm:
                respuesta_final = respuesta_llm
        
        # Limpiar respuesta final
        if not respuesta_final:
            respuesta_final = "No pude procesar tu consulta. Intenta reformularla."
        
        # Agregar a memoria
        self.memoria.agregar("assistant", respuesta_final)
        
        print("="*80 + "\n")
        return respuesta_final


# =============================================================================
# INTERFAZ DE CONSOLA
# =============================================================================

def main():
    """Función principal con manejo robusto de inicialización"""
    
    print("=" * 80)
    print("🥊 INICIALIZANDO ASISTENTE EVERLAST")
    print("=" * 80)
    
    # Cargar .env
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not github_token or not base_url:
        print("❌ Error: Variables de entorno no configuradas")
        print("   Verifica que .env tenga GITHUB_TOKEN y OPENAI_BASE_URL")
        sys.exit(1)
    
    print(f"\n✅ Configuración cargada")
    print(f"   • Token: ...{github_token[-8:]}")
    print(f"   • URL: {base_url}")
    
    # Crear agente
    print("\n🔧 Inicializando agente...")
    try:
        agente = AgenteEverlast(github_token, base_url)
    except Exception as e:
        print(f"❌ Error al crear agente: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ ASISTENTE LISTO")
    print("=" * 80)
    print("\nComandos:")
    print("  • 'salir' - Terminar")
    print("  • 'limpiar' - Borrar memoria")
    print("  • 'historial' - Ver conversación")
    print("\n💡 Ejemplos:")
    print("  • ¿Qué guantes recomiendas?")
    print("  • Calcula 15% descuento en $50000")
    print("  • ¿Cuál es la política de devolución?")
    print("=" * 80 + "\n")
    
    # Loop principal
    while True:
        try:
            user_input = input("👤 Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 ¡Hasta luego!")
            break
        
        # Comandos especiales
        if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n👋 ¡Gracias por usar Everlast Chile!")
            break
        
        elif user_input.lower() in ['limpiar', 'clear', 'reset']:
            agente.memoria.limpiar()
            print("\n🧹 Memoria limpiada\n")
            continue
        
        elif user_input.lower() in ['historial', 'history', 'h']:
            print("\n📜 HISTORIAL:")
            print("=" * 80)
            historial = agente.memoria.historial
            if not historial:
                print("(Vacío)")
            else:
                for i, msg in enumerate(historial, 1):
                    emoji = "👤" if msg["role"] == "user" else "🤖"
                    print(f"\n{i}. {emoji} {msg['role'].upper()}:")
                    print("-" * 80)
                    print(msg['content'][:300] + ("..." if len(msg['content']) > 300 else ""))
            print("=" * 80 + "\n")
            continue
        
        elif not user_input:
            continue
        
        # Procesar consulta
        try:
            respuesta = agente.procesar(user_input)
            print(f"\n🤖 Everlast: {respuesta}\n")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación cancelada")
            continue
        
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("   Intenta con otra pregunta\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)