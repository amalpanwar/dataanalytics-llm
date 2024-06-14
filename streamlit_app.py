#pip install lida[transformers]
from lida import Manager, TextGenerationConfig , llm
from dotenv import load_dotenv
import os
#import openai
from PIL import Image
from io import BytesIO
import base64
import streamlit as st
lida=Manager(text_gen=llm(provider="hf", model="HuggingFaceH4/zephyr-7b-beta", device_map="auto"))
hf_config = TextGenerationConfig(n=1,temperature=0.5, max_tokens=650, use_cache=True)

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))
menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "EDA"])
if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your excel", type="xlsx")
    if file_uploader is not None:
        path_to_save = "/content/player_data_unq.xlsx"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("player_data_unq.xlsx", summary_method="llm", textgen_config=hf_config)
        st.write(summary)
        goals = lida.goals(summary, n=2, textgen_config=hf_config)
        for goal in goals:
            st.write(goal)
        i = 0
        library = "Matplotlib"
        hf_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=hf_config, library=library)
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)
if menu == "EDA":
    st.subheader("Query your Data to Generate Visuals")
    file_uploader = st.file_uploader("Upload your excel", type="xlsx")
    if file_uploader is not None:
        path_to_save = "/content/player_data_unq.xlsx"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Query your Data to Generate Visuals", height=200)
        if st.button("Generate Visuals"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                lida = Manager(text_gen=llm(provider="hf", model="HuggingFaceH4/zephyr-7b-beta", device_map="auto"))
                hf_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize("player_data_unq.xlsx", summary_method="llm", textgen_config=hf_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=hf_config)
                charts[0]
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)
