from langchain_community.chat_models import BedrockChat
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1"
)

# Claude 3 model (on-demand compatible)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Use BedrockChat instead of Bedrock
llm = BedrockChat(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9}
)

def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=['language', 'user_text'],
        template="You are a helpful AI assistant. You respond in {language}.\n\nUser: {user_text}\nAI:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain({'language': language, 'user_text': user_text})

# Streamlit UI
st.title("Bedrock Claude 3 Chatbot")

language = st.sidebar.selectbox("Language", ["English", "Spanish", "Tamil"])
user_text = st.sidebar.text_area("What is your question?", max_chars=100)

if user_text:
    response = my_chatbot(language, user_text)
    st.write(response['text'])
