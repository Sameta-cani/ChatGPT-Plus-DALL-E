import streamlit as st
import openai
import func
import diffusers
import cv2
import PIL
import numpy as np
import model
# import torch

openai.api_key = st.secrets["api_key"]

st.title("ChatGPT Plus DALL-E")
# st.write(torch.__version__)
st.image('./증명사진.jpg')
with st.form(key="form"):
    user_input = st.text_input(label="Prompt")
    size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"])
    submit = st.form_submit_button(label="Submit")

prompt = "masterpiece, best quality, ultra-detailed, illustration, school uniform, scarf, gymnasium"
negative_prompt = "lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
num_steps = 20
guidance_scale = 7
seed = 3467120481370323442

image, canny_image, out_image = model.img2img("증명사진.jpg", prompt, negative_prompt, num_steps, guidance_scale, seed)
st.image(out_image)

if submit and user_input:
    gpt_prompt = [{
        "role": "system",
        "content": "Imagine the detail appeareance of the input. Response it shortly around 20 words."
    }]

    gpt_prompt.append({
        "role": "user",
        "content": user_input
    })
    with st.spinner(text="Waiting for ChatGPT..."):
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_prompt
        )

    prompt = gpt_response["choices"][0]["message"]["content"]
    st.write(prompt)

    with st.spinner(text="Waiting for DALL-E..."):
        dalle_response = openai.Image.create(
            prompt=prompt,
            size=size
        )
    
    st.image(dalle_response["data"][0]["url"])
