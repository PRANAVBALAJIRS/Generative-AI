import getpass
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-pro",
    temperature = 0,
    max_tokens = None,
    timeout = None,
    max_retries = 2
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}"
        ),
        ("human", "{input}"),
    ]
)

st.title('Langchain Demo with Gemini(Language Translator)')
input_text = st.text_input("Write the sentence in English and it will be translated to Tamil")

output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke(
        {
            "input_language": "English",
            "output_language": "Tamil",
            "input": input_text,
        }
    ))