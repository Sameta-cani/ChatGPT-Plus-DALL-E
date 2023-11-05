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
      print(type(img_file))
      print(img_file.name)
      print(img_file.size)
      print(img_file.type)

      save_uploaded_file('image', img_file)

      st.image(f'image/{img_file.name}')

print(model.img2img)

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
