import vertexai
import streamlit as st
import os
from vertexai.preview.generative_models import(
    GenerationConfig,
    GenerativeModel
)
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("project_id")
project_region = os.getenv("region")

vertexai.init(project=project_id, location=project_region)

model = GenerativeModel("gemini-2.5-pro")

def user_interfaces():
    st.set_page_config("VertexAI Demo")
    st.header("Gemini on VertexAI")

    user_question = st.text_input("Ask me anything ...")

    if user_question:
        response = model.generate_content(user_question, stream=True)

        for res in response:
            st.write(res.text, end="")

if __name__=="__main__":
    user_interfaces()