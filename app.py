import gradio as gr
import pandas as pd
import os
from groq import Groq
import google.generativeai as genai

# ==================== CARGAR CATÁLOGO ====================
df = pd.read_csv("catalogo_footloose_limpio.csv")
catalogo_texto = df[["modelo", "marca", "categoria", "genero", "precio"]].to_string(index=False)

# ==================== CLIENTES ====================
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def chat(mensaje, historial):
    if not mensaje or not mensaje.strip():
        return "", historial, historial

    # Prompt más limpio y corto
    system_prompt = f"""Eres un asistente amigable experto en zapatillas.
Usa este catálogo real para responder:
{catalogo_texto}

Responde siempre en español, corto, claro y útil.
Si no tienes la información, dilo con honestidad."""

    # Construir mensajes correctamente
    messages = [{"role": "system", "content": system_prompt}]

    for user_msg, assistant_msg in historial:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": mensaje})

    respuesta = "Lo siento, estoy teniendo problemas de conexión. Inténtalo de nuevo en unos segundos."

    # === INTENTAR GROQ PRIMERO ===
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=350,
            temperature=0.7
        )
        respuesta = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")

    # === FALLBACK A GEMINI ===
    if len(respuesta) < 10 or "Lo siento" in respuesta:
        try:
            # Convertir a formato simple para Gemini
            gemini_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = gemini_model.generate_content(gemini_prompt)
            respuesta = response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            respuesta = "❌ Ambos servicios están ocupados. Por favor intenta de nuevo más tarde."

    # Actualizar historial
    historial.append([mensaje, respuesta])

    return "", historial, historial


# ==================== INTERFAZ ====================
with gr.Blocks(title="👟 Bot Zapatillas") as demo:
    gr.Markdown("# 👟 Asistente de Zapatillas\n> Pregúntame por modelos, precios y marcas")

    historial_state = gr.State([])
    chatbot = gr.Chatbot(height=500, label="Chat")

    msg = gr.Textbox(
        placeholder="Ej: ¿Tienen Nike para mujer en talla 38?",
        label="Tu pregunta"
    )

    with gr.Row():
        enviar = gr.Button("Enviar 🚀", variant="primary")
        limpiar = gr.Button("🗑️ Limpiar")

    # Conexiones
    enviar.click(chat, inputs=[msg, historial_state], outputs=[msg, historial_state, chatbot])
    msg.submit(chat, inputs=[msg, historial_state], outputs=[msg, historial_state, chatbot])

    limpiar.click(
        lambda: ([], []),
        inputs=None,
        outputs=[historial_state, chatbot]
    )

demo.launch()
