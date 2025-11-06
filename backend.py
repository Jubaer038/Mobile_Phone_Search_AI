import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

CHROMA_PATH = "chroma_db"
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # .env file এ add করো

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def text_to_chunks(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_or_load_chroma(chunks, collection_name="default_collection"):
    embeddings = get_embeddings()
    os.makedirs(CHROMA_PATH, exist_ok=True)
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name
    )
    vectordb.persist()
    return vectordb

def create_llm(model_repo="google/flan-t5-large"):
    if not HF_API_KEY:
        raise ValueError("HuggingFace API key not found in .env")
    return HuggingFaceHub(
        repo_id=model_repo,
        huggingfacehub_api_token=HF_API_KEY,
        model_kwargs={"temperature":0.2,"max_length":512}
    )

def create_conversational_chain(vectordb, llm=None):
    if llm is None:
        llm = create_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordb.as_retriever(search_kwargs={"k":3})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return chain, memory
