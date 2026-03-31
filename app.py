import gradio as gr
import pandas as pd
import os
from groq import Groq
import google.generativeai as genai

# Cargar catálogo una sola vez
df = pd.read_csv("catalogo_footloose_limpio.csv")
catalogo_texto = df[["modelo", "marca", "categoria", "genero", "precio"]].to_string(index=False)

# Clientes API
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def chat(mensaje, historial):
    if not mensaje or not mensaje.strip():
        return "", historial, historial

    # Prompt corto
    prompt = f"""Eres un asistente amigable de zapatillas. 
Catálogo: {catalogo_texto}

Responde en español, corto y útil.

Historial anterior:
{"\n".join([f"Usuario: {u}\nAsistente: {a}" for u,a in historial])}

Usuario: {mensaje}
Asistente:"""

    respuesta = "Lo siento, estoy teniendo problemas ahora. Inténtalo de nuevo."

    # Intentar Groq primero
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        respuesta = completion.choices[0].message.content
    except Exception:
        # Fallback a Gemini
        try:
            response = gemini_model.generate_content(prompt)
            respuesta = response.text
        except Exception as e:
            respuesta = f"❌ Error temporal: {str(e)[:100]}"

    historial.append([mensaje, respuesta])
    return "", historial, historial


# Interfaz
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")

    historial_state = gr.State([])
    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(placeholder="Ej: ¿Tienen Nike para mujer en talla 38?", label="Tu pregunta")

    with gr.Row():
        gr.Button("Enviar 🚀", variant="primary").click(
            chat, [msg, historial_state], [msg, historial_state, chatbot]
        )
        gr.Button("Limpiar").click(
            lambda: ([], []), None, [historial_state, chatbot]
        )

    msg.submit(chat, [msg, historial_state], [msg, historial_state, chatbot])

demo.launch()
