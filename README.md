# Using-paligemma-to-make-image-captions
In this program you will find how to export a Google “paligemma” model to interact with the image, you will be able to ask whatever you want to know about the image. This program use streamlit as GUI. 

---
## Requirements 

I recomend to install the following libraries in the following order:

- python = 3.9
- pip install git+https://github.com/huggingface/transformers
- pip install transformers
- conda install pytorch=2.2.2 torchvision==0.17.2 torchaudio===2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install anaconda::ipython
- pip install streamlit
  
---
## Code
### Importing libraries 
```python
import streamlit as st
import os
import torch
import PIL.Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import requests
import tempfile
import urllib.parse as parse

```
### Defining fuctions to load the image
```python
headers = {
  'Authorization': 'This is only for image from wikimedia commons',
  'User-Agent': 'user for wikimedia commons'
}

# Verify url
def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# Load an image
def load_image(image_path):
    if check_url(image_path):
        respuesta = requests.get(image_path, headers=headers, stream=True)#.raw
        return PIL.Image.open(requests.get(image_path, headers=headers, stream=True).raw)
    elif os.path.exists(image_path):
        return PIL.Image.open(image_path)
```

### Fuction to load the pre trained model
```python
@st.cache_resource
def load_model():
    # Assign available GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Aquí debes crear un token en Hugging Face para usar el modelo preentrenado
    access_token = "here you create a token in hugging face for access to the models"
    model_id = "google/paligemma-3b-mix-224"

    # Cargar el modelo en el dispositivo adecuado
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        revision="bfloat16",
        token=access_token
    ).eval()

    # Cargar el procesador de imágenes
    image_processor = AutoProcessor.from_pretrained(model_id, token=access_token)

    return model, image_processor
```
### Main function
```python
def main():
    
    st.title("DESCRIPCION OF THE IMAGE")
    # Llama a la función cacheada para cargar el modelo
    model, image_processor = load_model()
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

    # Mostrar la imagen si se ha subido un archivo
    if uploaded_file is not None:
        image_show = PIL.Image.open(uploaded_file)
        st.image(image_show, caption='Imagen cargada por el usuario', use_column_width=True)
        
        # Guardar el archivo en un directorio temporal
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name
            image = load_image(temp_path)

        prompt = st.text_input('How can i help you?')
        model_inputs = image_processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, repetition_penalty=1.10, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]
            decoded = image_processor.decode(generation, skip_special_tokens=True)
            CAPTION = decoded  
        
        st.write(CAPTION)    
    

if __name__== '__main__':
    main()

```






