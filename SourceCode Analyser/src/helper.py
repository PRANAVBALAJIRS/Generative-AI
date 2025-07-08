from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings

# To clone github repo
def repo_ingestion(repo_url):
    if os.path.exists("repo"):
        os.system("rm -rf repo")
    os.makedirs("repo", exist_ok=True)
    Repo.clone_from(repo_url, to_path="repo/")

# Loading repo as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py", ".ipynb", ".md", ".txt"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    return documents


def text_splitter(documents):
    document_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                 chunk_size = 500,
                                                                 chunk_overlap = 20)
    
    texts_chunks = document_splitter.split_documents(documents)
    return texts_chunks

def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings