from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient
import os

load_dotenv("os.env") 

st.set_page_config(page_title="Chatbot", page_icon=":robot:")

API_KEY = os.getenv("Token")  
MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"  

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

    context = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[-10:]]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=context,  
        max_tokens=500
    )

    generated_text = completion.choices[0].message["content"]
    st.session_state.messages.append({"role": "assistant", "content": generated_text})

    with st.chat_message("assistant"):
        st.markdown(generated_text)
