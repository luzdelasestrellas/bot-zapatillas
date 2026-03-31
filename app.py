import gradio as gr
import pandas as pd
import os
from groq import Groq

# ==================== CARGAR CATÁLOGO ====================
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
df = pd.read_csv("catalogo_footloose_limpio.csv")   # ← nombre correcto del archivo
catalogo_texto = df[["modelo", "marca", "categoria", "genero", "precio"]].to_string(index=False)

# ==================== CLIENTE GROQ (MÁS BARATO) ====================
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat(mensaje, historial):
    if not mensaje.strip():
        return "", historial, historial

    # Mensajes para Groq
    mensajes = [
        {
            "role": "system",
            "content": f"""Eres un asistente amigable y experto en zapatillas.
Tienes este catálogo real:
{catalogo_texto}

Responde SIEMPRE en español, corto, claro y amigable.
Si no encuentras el modelo o la información, dímelo con honestidad."""
        }
    ]

    # Agregar historial
    for user_msg, assistant_msg in historial:
        mensajes.append({"role": "user", "content": user_msg})
        mensajes.append({"role": "assistant", "content": assistant_msg})

    mensajes.append({"role": "user", "content": mensaje})

    # Llamada a Groq con modelo MÁS BARATO y con más límites gratis
    try:
        respuesta = client.chat.completions.create(
            model="llama-3.1-8b-instant",     # ←←← ESTE ES EL MÁS BARATO Y CON MÁS LÍMITE
            messages=mensajes,
            max_tokens=500,
            temperature=0.7
        ).choices[0].message.content

    except Exception as e:
        respuesta = f"❌ Error: {str(e)}\n\nInténtalo de nuevo en unos segundos."

    # Actualizar historial
    historial.append([mensaje, respuesta])

    return "", historial, historial


# ==================== INTERFAZ GRADIO ====================
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")

    historial_state = gr.State([])
    chatbot = gr.Chatbot(height=400, label="Chat")
    
    msg = gr.Textbox(
        placeholder="Ej: ¿Qué zapatillas tienen para mujer?",
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
