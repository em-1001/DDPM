# Reference : https://www.youtube.com/watch?v=ZBKpAp_6TGI

import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using Device: {DEVICE}")

tokenizer = CLIPTokenizer("./tokenizer_vocab.json", merges_file="./tokenizer_merges.txt")
model_file = "/content/drive/MyDrive/ddpm/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
# models = preload_models_from_standard_weights(model_file, DEVICE)


## TEXT TO IMAGE
prompt = "A cat with blue eyes, highly detailed, ultra sharp, cinematic, 8k resolution"
uncond_prompt = "" # You can use it as a negative prompt
do_cfg = True
cfg_scale = 7


## IMAGE TO IMAGE
input_image = None
image_path = "./dog.jpg"
# input_image = Image.open(image_path)
strength = 0.9

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt = prompt,
    uncond_prompt = uncond_prompt,
    input_image = input_image,
    strength = strength,
    do_cfg = do_cfg,
    cfg_scale = cfg_scale,
    sampler_name = sampler,
    n_inference_steps = num_inference_steps,
    seed = seed,
    models = models,
    device = DEVICE,
    idle_device = "cpu",
    tokenizer = tokenizer
)


Image.fromarray(output_image)
