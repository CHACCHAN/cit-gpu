import os
import torch
import random, string
from diffusers import DiffusionPipeline

def randomname(n):
    randlist = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return "".join(randlist)

prompt = "A high-end commercial DSLR fashion portrait of a beautiful 20-year-old Japanese woman, short bob haircut with heavy bangs, pale skin, soft pink blush on cheeks, monolids eyes, single eyelid, elegant makeup, wearing a white luxury silk shirt, highly detailed skin texture, 8k resolution"

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=3.0,
    width=1024,
    height=1024,
).images[0]

os.makedirs("images", exist_ok=True)

image_name = randomname(16)
image_path = f"images/{image_name}.png"
image.save(image_path)

print(f"save as {image_name}")