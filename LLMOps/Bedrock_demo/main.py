from langchain_community.llms import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")

bedrock = boto3.client(
    service_name = "bedrock-runtime",
    region_name = region_name,
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key
)

model_id = "mistral.mistral-7b-instruct-v0:2"

llm = Bedrock(
    model_id = model_id,
    client = bedrock,
    model_kwargs = {"temperature": 0.9}
)

def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables = ["language", "user_text"],
        template = "You are a chatbot. You are in {language}.\n\n{user_text}"
    )
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'user_text': user_text})

    return response

st.title("Bedrock Chatbot Demo")

language = st.sidebar.selectbox("Language", ["english", "spanish", "hindi"])

if language:
    user_text = st.sidebar.text_area(label="What is your question?", max_chars=100)

if user_text:
    response = my_chatbot(language, user_text)
    st.write(response['text'])