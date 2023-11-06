import streamlit as st
import openai
import func
import diffusers
import cv2
import PIL
import numpy as np
import os
import model

# 파일 업로드 함수
def save_uploaded_file(directory, file):
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(os.path.join(directory, file.name), 'wb') as f:
		f.write(file.getbuffer())
	return st.success('파일 업로드 성공')

openai.api_key = st.secrets["api_key"]

st.title("ChatGPT Plus DALL-E")

img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
if img_file is not None:
    st.write(type(img_file))
    st.write(img_file.name)
    st.write(img_file.size)
    st.write(img_file.type)

    save_uploaded_file('image', img_file)

    st.image(f'image/{img_file.name}')

    prompt = "masterpiece, best quality, ultra-detailed, illustration, school uniform, scarf, gymnasium"
    negative_prompt = "lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
    num_steps = 20
    guidance_scale = 7
    seed = 3467120481370323442

    image, canny_image, out_image = model.img2img(f'image/{img_file.name}', prompt, negative_prompt, num_steps, guidance_scale, seed)

    st.image(out_image)

''''
### original form ###

with st.form(key="form"):
    user_input = st.text_input(label="Prompt")
    size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"])
    submit = st.form_submit_button(label="Submit")

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
'''