import gradio as gr
import pandas as pd
import os
from groq import Groq
import google.generativeai as genai

# ==================== CARGAR CATÁLOGO ====================
df = pd.read_csv("catalogo_footloose_limpio.csv")
catalogo_texto = df[["modelo", "marca", "categoria", "genero", "precio"]].to_string(index=False)

# ==================== CLIENTES API ====================
# Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Gemini (cambia tu clave aquí)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')   # modelo rápido y con buen límite gratis

# ==================== FUNCIÓN CHAT CON FALLBACK ====================
def chat(mensaje, historial):
    if not mensaje.strip():
        return "", historial, historial

    # Prompt del sistema (más corto para ahorrar tokens)
    system_prompt = f"""Eres un asistente amigable de zapatillas. 
Tienes este catálogo real: {catalogo_texto}
Responde SIEMPRE en español, corto, claro y útil.
Si no encuentras algo, dilo con honestidad."""

    # Construir mensajes con historial
    mensajes = [{"role": "user", "content": system_prompt}]

    for user_msg, assistant_msg in historial:
        mensajes.append({"role": "user", "content": user_msg})
        mensajes.append({"role": "assistant", "content": assistant_msg})

    mensajes.append({"role": "user", "content": mensaje})

    respuesta = None

    # === 1. Intentar primero con Groq ===
    try:
        respuesta = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=mensajes,
            max_tokens=400,
            temperature=0.7
        ).choices[0].message.content
    except Exception as e:
        print(f"Groq falló: {e}")   # para ver en logs

    # === 2. Si Groq falla, usar Gemini como respaldo ===
    if not respuesta:
        try:
            gemini_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in mensajes])
            response = gemini_model.generate_content(gemini_prompt)
            respuesta = response.text
        except Exception as e2:
            respuesta = "❌ Lo siento, ambos servicios están ocupados ahora. Inténtalo en unos minutos."

    # Actualizar historial
    historial.append([mensaje, respuesta])

    return "", historial, historial


# ==================== INTERFAZ GRADIO ====================
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")

    historial_state = gr.State([])
    chatbot = gr.Chatbot(height=450, label="Chat")
    
    msg = gr.Textbox(
        placeholder="Ej: ¿Qué zapatillas tienen para mujer en talla 38?",
        label="Tu pregunta"
    )

    with gr.Row():
        enviar = gr.Button("Enviar 🚀", variant="primary")
        limpiar = gr.Button("🗑️ Limpiar")

    # Conexiones
    msg.submit(chat, [msg, historial_state], [msg, historial_state, chatbot])
    enviar.click(chat, [msg, historial_state], [msg, historial_state, chatbot])

    limpiar.click(
        lambda: ([], [], []),
        None,
        [historial_state, chatbot, msg],
        queue=False
    )

demo.launch()
