import os
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import load_embedding
from dotenv import load_dotenv
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAIAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
persist_directory = "data"
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm = llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"K":8}), memory=memory)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    if request.method == 'POST':
        user_input = request.form['question']
        
        repo_ingestion(user_input)
        os.system("python store_index.py")

        # ⚠️ Reload the updated DB
        global vectordb, qa, memory
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
            memory=memory
        )

        return jsonify({"response": "Repository Ingested and Vector DB updated."})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

        global memory, qa
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
            memory=memory
        )
        return "🧹 Chat cleared and memory reset."

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])

if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)