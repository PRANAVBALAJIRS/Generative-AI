import os
import openai
from dotenv import load_dotenv
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader('articles').load_data()

index = GPTVectorStoreIndex.from_documents(documents)

index.storage_context.persist() 