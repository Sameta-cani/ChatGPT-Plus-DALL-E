from model import img2img
import numpy as np
from PIL import Image

prompt = "masterpiece, best quality, ultra-detailed, illustration, school uniform, scarf, gymnasium"
negative_prompt = "lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
num_steps = 20
guidance_scale = 7
seed = 3467120481370323442

image, canny_image, out_image = img2img("증명사진.jpg", prompt, negative_prompt, num_steps, guidance_scale, seed)

out_image.save("02_result.png")
Image.fromarray(np.concatenate([image.resize(out_image.size), canny_image.resize(out_image.size), out_image], axis=1))