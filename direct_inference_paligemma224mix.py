#In this program you will find a image captioning program that use images
#from the wikimedia commons API.

#____________General libraries______________________________
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from transformers import pipeline
import requests
import argparse

#______________libraries for Image Captioning______________________
# Backend
import torch
# Image Processing
#from PIL import Image
import PIL.Image
# Transformer and pre-trained Model
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# Managing loading processing
from tqdm import tqdm
# Assign available GPU


#___________________Libraries for the progress bar_____________________
import time
from progressbar import ProgressBar

#___________________________________________________________________

#____________________Loading the model______________________________

#Here you have to create a token in hugging face to use the pre trained model 

access_token = "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" 

#model_id = "google/paligemma-3b-pt-224"
model_id = "google/paligemma-3b-mix-224"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16


model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
    token=access_token
).eval()




image_processor = AutoProcessor.from_pretrained(model_id, token=access_token)



# ___________Accesssing images from the web or the files___________

#Headers for wikimedia

headers = {
  'Authorization': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI1ZGNmOWIxYTBlNDFmZTE3ZjYxMjYwMTY3NDM2ZmNjYSIsImp0aSI6ImZmMDVkYjIxOWFlMTJhNWRiOWY0ZTU4NTFlNWI3NTEzY2QwM2JhYTkyM2NmZjNkZTdjMWJlYTc4YTUzODg2ZGFkMWZjNjY5Y2EyZGY4Yjc4IiwiaWF0IjoxNzEzODE1Mjk4LjIyNzIwMiwibmJmIjoxNzEzODE1Mjk4LjIyNzIwNSwiZXhwIjozMzI3MDcyNDA5OC4yMjU1OSwic3ViIjoiNzUyMjAwOTQiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIiwiY3JlYXRlZWRpdG1vdmVwYWdlIl19.klqz-hpRMnLhqORQAOY7QNxash20FAM9wX3IxsV7_QtLRBLx83VUIb_22oJG9_w-gi0A_cQ9fw8GCKp4Hfp0Z7fJsT9ragbs2bJp6o9ztowx4BrN32QhPEXAU9C-pjC6WsbpnFUzKRnZwz3_Kj4NxCXVQMsB6kKhyjTX-KutdoAE7YVvl-g13AviUhFjitNMVW7KZIJkK9hd1N2GI5gtc75gkjvDSRjr1pTubJXl8lzqWfpi9IjovoujhKe_0N8_i0dOlwLoRhcNaWoTJ22O7o4Fcku4aWFgnlLJF7Q0ZjVsHiCr9h1_OX7xlduApuj0m6qaCokU2PEwKdgfEKHRm1V9mjY7ANl3BJrT9JDMo_BvJiKkuhheyJY6RENEqLwvWinfW87aWPdp-9kn07i6o-vytLnEC093YdwYARdvZhftUHgdsmE0LsMWBWoKIUcux8FXcRtgTKCZ3AHNJ2ik3Gu5vQWzl4jKd6cKuAOp-jvgLkuUUR-ateSFrmx9gyPhjWVPkl4jSekGqRHYJE3no8yAbk4v5yYjRvfWbvYKKLtmQ4GuLMhLxfJX6WOfnzzDHpq2LXKjUpIMhdFxcpebVEKtY7mPfoeZCvYSkMZQB10Kbk4XiWUPUJ125xKr4A3r4Ai8Nxyk-DbJzxjo-POfUSMm9UO_2WLWCIYbCrEahwo',
  'User-Agent': 'MithozZfg'
}

import urllib.parse as parse
import os
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
    

#______________________Open the program from terminal_____________________

# Generate new caption from input image
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument('--category', help="Category you want to work")
parser.add_argument('--iteration', type=int, default=10, help="Number of the images to work")

CATEGORY = parser.parse_args().category
ITERATIONS = parser.parse_args().iteration 




if __name__=='__main__':
    

    #________________________________________________________________________________
    #________________________________________________________________________________
    #_________A continuación se generarán los CAPTIONS de cada imagen________________
    #________________________________________________________________________________
    #________________________________________________________________________________


     
    
    #imagen_path = '/home/mitos/Documentos/AVANCE JULIO/exponiendo.jpg'
    imagen_path = "https://estaticosgn-cdn.deia.eus/clip/37ef816f-bb07-46b0-8c6d-6887ba5b8ee6_16-9-discover-aspect-ratio_default_0.jpg"
    
            
    #se descarga en catche la imagen
    image = load_image(imagen_path)
    print("pasa3")

    cont = 0

    while cont<=10:
        prompt = input("How can i help you?: ")
        if prompt == "exit":
            cont = 10

        model_inputs = image_processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, repetition_penalty=1.10, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]
            decoded = image_processor.decode(generation, skip_special_tokens=True)
            CAPTION = decoded  
            
        cont += 1
        print(CAPTION)
        
        
    
    
    print("FIN")