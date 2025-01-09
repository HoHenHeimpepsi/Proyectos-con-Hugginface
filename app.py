import streamlit as st
from huggingface_hub import InferenceClient
import os

st.set_page_config(page_title="Chatbot", page_icon=":robot:")

# Configuración de la API
API_KEY = os.getenv("key")
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"

client = InferenceClient(api_key=API_KEY)

st.title("Chatbot")
st.write("Escribe un mensaje y presiona Enter para obtener una respuesta.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe un mensaje..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-10:]])

    response = client.text_generation(
        model=MODEL_NAME,
        prompt=f"{context}\nAsistente:",
        max_new_tokens=500
    )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
