import streamlit as st
from PIL import Image
from io import BytesIO
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

model_audio = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

model_audio.tie_weights()

image_client = InferenceClient("strangerzonehf/Flux-Midjourney-Mix2-LoRA", token=os.getenv('HF_KEY'))

# ---------- Interfaz con Streamlit ----------
st.title("Chatbot de Audio a Imagen")

audio_file = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.subheader("🎙️ ¡Archivo de audio subido correctamente!")
    
    st.audio(audio_file, format="audio/wav")
    
    with st.spinner("Procesando audio..."):
        try:
            
            audio_bytes = audio_file.read()

            audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=processor.feature_extractor.sampling_rate)
            
            conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_bytes}]}]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            inputs = processor(text=text_prompt, audios=[audio_data], return_tensors="pt", padding=True)
            generate_ids = model_audio.generate(**inputs, max_length=256)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            st.success(f"📝 Texto generado: {generated_text}")

            if generated_text:
                with st.spinner("Generando imagen..."):
                    image = image_client.text_to_image(generated_text)
                    st.image(image, caption="Imagen generada", use_column_width=True)

                    img_path = "imagen_generada.png"
                    image.save(img_path)
                    st.success(f"✅ Imagen guardada como: {img_path}")
            else:
                st.error("No se generó texto a partir del audio.")
        except Exception as e:
            st.error(f"Error en el procesamiento del audio: {str(e)}")
else:
    st.warning("Por favor sube un archivo de audio.")
