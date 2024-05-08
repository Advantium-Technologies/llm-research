from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyMuPDFLoader


import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import  CharacterTextSplitter

import chromadb
import ollama
from langchain_community.llms import Ollama



client = chromadb.Client()
client.heartbeat()

st.write("#Chat with DFCCIL PDF")

if "model" not in st.session_state:
    st.session_state["model"] = ""

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Choose your model", models, key='model_selection')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Create a temporary file to write the bytes to
    with open("temp_pdf_file.pdf", "wb") as temp_file:
        temp_file.write(uploaded_file.read())
        st.session_state["pdf_file"] = 'temp_pdf_file.pdf'


#
# class PdfGpt():
#     def __init__(self, file_path):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#         chunks = text_splitter.split_documents(documents=PyMuPDFLoader(file_path=file_path).load())
#         embedding_model = HuggingFaceEmbeddings(
#             model_name="nomic-embed-text",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )

def get_llm_response(query):
    vectordb = load_chunk_persist_pdf()
    chain = load_qa_chain(Ollama(model=st.session_state['model']), chain_type="stuff")
    matching_docs = vectordb.similarity_search(query)
    print(f"matching doc {matching_docs}")
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


def create_agent_chain():
    chain = load_qa_chain(Ollama(model=st.session_state['model']), chain_type="stuff")
    return chain

@st.cache_resource
def load_chunk_persist_pdf() -> Chroma:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_documents(documents=PyMuPDFLoader(file_path=st.session_state['pdf_file']).load())
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("data_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        persist_directory="."
    )
    vectordb.persist()
    return vectordb


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to say to your PDF?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if len(prompt):
            with st.spinner("Generating response..."):
                message = st.write(get_llm_response(prompt))
                st.session_state["messages"].append({"role": "assistant", "content": message})



