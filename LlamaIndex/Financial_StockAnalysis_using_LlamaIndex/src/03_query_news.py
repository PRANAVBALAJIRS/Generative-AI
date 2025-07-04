import os
import openai
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

response = query_engine.query("Tell me about the new Google Supercomputers")
print(response)
