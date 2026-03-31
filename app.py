import gradio as gr
import pandas as pd
import os
from groq import Groq

# Cargar catálogo (solo una vez al iniciar el Space)
df = pd.read_csv("catalogo_footloose_limpio.csv")
catalogo_texto = df[["modelo","marca","categoria","genero","precio"]].to_string(index=False)

# Cliente de Groq (usa la clave que guardaste en Secrets)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat(mensaje, historial):
    if not mensaje.strip():                # ← evita mensajes vacíos
        return "", historial, historial

    # === 1. Construir mensajes para Groq (formato correcto) ===
    mensajes = [
        {"role": "system", "content": f"""Eres un asistente amigable y experto en zapatillas.
Tienes este catálogo real:
{catalogo_texto}

Responde SIEMPRE en español, corto, claro y amigable.
Si no encuentras algo, dímelo con honestidad."""}
    ]

    # Agregar historial anterior (ahora en formato Gradio: listas [user, assistant])
    for user_msg, assistant_msg in historial:   # ← aquí está la corrección principal
        mensajes.append({"role": "user", "content": user_msg})
        mensajes.append({"role": "assistant", "content": assistant_msg})

    mensajes.append({"role": "user", "content": mensaje})

    # === 2. Llamada a Groq con manejo de errores ===
    try:
        respuesta = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=mensajes,
            max_tokens=500,
            temperature=0.7
        ).choices[0].message.content
    except Exception as e:
        respuesta = f"❌ Error al conectar con Groq: {str(e)}\n\n¿Puedes intentarlo de nuevo?"

    # === 3. Actualizar historial en el formato correcto ===
    historial.append([mensaje, respuesta])   # ← lista de 2 elementos

    # Devolvemos 3 valores: limpiar textbox + actualizar State + actualizar Chatbot
    return "", historial, historial


# === Interfaz Gradio ===
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")

    historial_state = gr.State([])          # ← State que ahora sí se actualiza
    chatbot = gr.Chatbot(height=400, label="Chat")
    
    msg = gr.Textbox(placeholder="Ej: ¿Qué zapatillas tienen para mujer?", label="Tu pregunta")

    with gr.Row():
        enviar = gr.Button("Enviar 🚀", variant="primary")
        limpiar = gr.Button("🗑️ Limpiar")

    # Conexiones (ahora coinciden con los 3 outputs de la función)
    msg.submit(chat, [msg, historial_state], [msg, historial_state, chatbot])
    enviar.click(chat, [msg, historial_state], [msg, historial_state, chatbot])

    limpiar.click(lambda: ([], [], []), None, [historial_state, chatbot, msg], queue=False)

demo.launch()   # ← buena práctica
