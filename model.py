from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler, DDPMScheduler, EulerDiscreteScheduler, KDPM2DiscreteScheduler, EulerAncestralDiscreteScheduler, DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler, UniPCMultistepScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverSinglestepScheduler
import cv2
from PIL import Image
import numpy as np
# from diffusers import DDPMScheduler
import torch
# import random, sys

controlnet_model = "lllyasviel/sd-controlnet-canny"
sd_model = "Lykon/DreamShaper"

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

print(cv2.__version__)

def tmp(a, b):
    '''
    test
    '''
    return a + b

def img2img(img_path, prompt, negative_prompt, num_steps=20, guidance_scale=7, seed=0, low=100, high=200):
    image = load_image(img_path)

    np_image = np.array(image)

    canny_image = cv2.Canny(np_image, low, high)

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    out_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        image=canny_image
    ).images[0]

    return image, canny_image, out_image