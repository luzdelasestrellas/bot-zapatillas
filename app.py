import gradio as gr
import pandas as pd
from groq import Groq
import os

# cargar catálogo
df = pd.read_csv("catalogo.csv")
catalogo_texto = df[["modelo","marca","categoria","genero","precio"]].to_string(index=False)

# iniciar Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# función del chat
def chat(mensaje, historial):
    mensajes = [
        {"role": "system", "content": f"""Eres un asistente amigable de zapatillas.
Tienes este catálogo real:

{catalogo_texto}

Responde en español, corto y amigable.
Si no encuentras algo dilo con honestidad."""}
    ]

    # agregar historial previo en formato diccionario
    for turno in historial:
        mensajes.append({"role": "user",      "content": turno["content"] if isinstance(turno, dict) and turno.get("role") == "user" else (turno[0] if isinstance(turno, list) else "")})
        mensajes.append({"role": "assistant", "content": turno["content"] if isinstance(turno, dict) and turno.get("role") == "assistant" else (turno[1] if isinstance(turno, list) else "")})

    mensajes.append({"role": "user", "content": mensaje})

    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=mensajes,
        max_tokens=500
    ).choices[0].message.content

    historial.append({"role": "user",      "content": mensaje})
    historial.append({"role": "assistant", "content": respuesta})
    return "", historial

# interfaz
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")
    historial_state = gr.State([])
    chatbot = gr.Chatbot(height=300)
    msg = gr.Textbox(placeholder="¿Qué zapatillas tienen para mujer?", label="Tu pregunta")
    with gr.Row():
        enviar  = gr.Button("Enviar 🚀", variant="primary")
        limpiar = gr.Button("🗑️ Limpiar")
    msg.submit(chat, [msg, historial_state], [msg, chatbot])
    enviar.click(chat, [msg, historial_state], [msg, chatbot])
    limpiar.click(lambda: ([], []), None, [historial_state, chatbot], queue=False)

demo.launch()
