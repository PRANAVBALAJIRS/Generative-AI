import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the
question at the end but use atleast summarize with 250 words with detailed explanations.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1"
)

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

def get_documents():
    loader = PyPDFDirectoryLoader("data")
    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0", 
                client=bedrock,
                model_kwargs={'max_gen_len': 512})
    return llm

PROMPT = PromptTemplate(
    template = prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# def main():
#     st.set_page_config("RAG Demo")
#     st.header("End to end RAG Application")
#     user_question = st.text_input("Ask a question from the PDF Files")

#     with st.sidebar:
#         st.title("Update or Create Vector Store:")

#         if st.button("Store Vector"):
#             with st.spinner("Processing..."):
#                 docs = get_documents()
#                 get_vector_store(docs)
#                 st.success("Done")

#         if st.button("Send"):
#             with st.spinner("Processing..."):
#                 faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
#                 llm = get_llm()
#                 st.write(get_response_llm(llm, faiss_index, user_question))

def main():
    st.set_page_config("RAG Demo", page_icon="📄", layout="wide")
    st.title("📘 End-to-End RAG PDF QA App")

    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            border: 1px solid #d4d4d4;
            border-radius: 6px;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🧠 Vector Store Management")
        st.info("Load and store PDF vectors here before asking questions.")

        if st.button("📚 Create / Update Vector Store"):
            with st.spinner("📄 Loading and processing PDFs..."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("✅ Vector store successfully created!")

    with st.expander("❓ Ask a Question", expanded=True):
        user_question = st.text_input("🔍 Enter your question:", placeholder="E.g., What is the objective of this document?")
        send_button = st.button("🚀 Generate Answer")

    if send_button and user_question.strip() != "":
        with st.spinner("🤖 Generating answer..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm()
            try:
                response = get_response_llm(llm, faiss_index, user_question)
                st.success("✅ Answer generated!")
                st.markdown("### 📜 Answer")
                st.markdown(f"<div style='background-color:#f4f4f4; padding:15px; border-radius:8px'>{response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    elif send_button:
        st.warning("⚠️ Please enter a valid question.")


if __name__ == "__main__":
    main()