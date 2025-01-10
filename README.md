El modelo local usado es "Qwen2-Audio-7B-Instruct" para el procesamiento de audio en el proyecto llamado "SwitchModel"

Las instrucciones de uso del modelo local y un ejemplo de como funcionan se encuentran en ese proyecto.

Se instalaran las dependencias correspondientes entregadas en el archivo "requeriments.txt" de la siguiente manera.

"pip install -r requeriments.txt"

Tambien para descargar el modelo local en base a las instrucciones presentadas en la pagina de este se usa "pip install git+https://github.com/huggingface/transformers"

```python
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
```

Esto importa el modelo directamente en nuestro codigo.

para finalizar se asignan los roles en base a lo que uno necesite. En el caso de la interaccion entre modelos 
como en el proyecto "SwitchModel" 

```python
conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_bytes}]}]
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
```

Se buscaba que el modelo solo generara un texto por lo que en mi caso se tiene solo el rol "user".
