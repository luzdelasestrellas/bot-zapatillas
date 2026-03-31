import gradio as gr
import pandas as pd
import os
from groq import Groq
import google.generativeai as genai

# Cargar catálogo
df = pd.read_csv("catalogo_footloose_limpio.csv")
catalogo_texto = df[["modelo", "marca", "categoria", "genero", "precio"]].to_string(index=False)

# Clientes
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')   # modelo correcto y estable

def chat(mensaje, historial):
    if not mensaje or not mensaje.strip():
        return "", historial, historial

    # Prompt más corto para ahorrar tokens
    system_prompt = f"""Eres un asistente amigable de zapatillas. Usa este catálogo real:
{catalogo_texto}

Responde en español, corto y útil. Si no sabes, dilo honestamente."""

    # Construir historial
    mensajes = [system_prompt]
    for user_msg, assistant_msg in historial:
        mensajes.append(f"Usuario: {user_msg}")
        mensajes.append(f"Asistente: {assistant_msg}")
    mensajes.append(f"Usuario: {mensaje}")

    respuesta = "Lo siento, ambos servicios están ocupados. Inténtalo de nuevo en unos segundos."

    # Primero Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "\n".join(mensajes)}],
            max_tokens=400,
            temperature=0.7
        )
        respuesta = completion.choices[0].message.content
    except Exception as e:
        print("Groq falló:", str(e))

    # Si Groq falla, usar Gemini
    if "Lo siento" in respuesta or len(respuesta) < 10:
        try:
            full_prompt = "\n".join(mensajes)
            response = gemini_model.generate_content(full_prompt)
            respuesta = response.text
        except Exception as e:
            print("Gemini también falló:", str(e))
            respuesta = "❌ Error temporal. Por favor intenta de nuevo."

    historial.append([mensaje, respuesta])

    return "", historial, historial


# Interfaz
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")

    historial_state = gr.State([])
    chatbot = gr.Chatbot(height=450, label="Chat")
    
    msg = gr.Textbox(placeholder="Ej: ¿Tienen zapatillas Nike para mujer?", label="Tu pregunta")

    with gr.Row():
        enviar = gr.Button("Enviar 🚀", variant="primary")
        limpiar = gr.Button("🗑️ Limpiar")

    msg.submit(chat, [msg, historial_state], [msg, historial_state, chatbot])
    enviar.click(chat, [msg, historial_state], [msg, historial_state, chatbot])

    limpiar.click(lambda: ([], [], []), None, [historial_state, chatbot, msg], queue=False)

demo.launch()
