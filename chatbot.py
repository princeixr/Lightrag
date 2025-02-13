from openai import OpenAI
import openai
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import bs4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader 
import textwrap
import pickle 
from langchain_community.vectorstores import FAISS
import pandas as pd 
import faiss 
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def process_pdf(doc_path):
    """
    Reads a PDF file and extracts text using PyPDFLoader.
    """
    try:
        # Use PyPDFLoader to load the document
        loader = PyPDFLoader(doc_path)
        document = loader.load()

        # Combine all pages into a single string
        document_text = " ".join([page.page_content for page in document])
        return document_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    

def chunking_document(document_string):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(document_string)
    return doc_chunks 

def get_response(query, retriever, llm):

    context = retriever.invoke(query)
    
    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", query),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response['answer']

# New RAG Architecture
def light_rag(doc_path):

    WORKING_DIR = "./dickens"

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
    )

    doc_text = process_pdf(doc_path)

    # with open(doc_path, "r", encoding="utf-8") as f:
    rag.insert(doc_text)
    # # Perform naive search
    # print(
    #     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
    # )

    # # Perform local search
    # print(
    #     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
    # )

    # # Perform global search
    # print(
    #     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
    # )

    # # Perform hybrid search
    # print(
    #     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
    # )

    
    return rag

# Set app layout to full screen
st.set_page_config(layout="wide")

st.title("Document Chatbot")
uploaded_file = st.file_uploader("Upload a PDF Document", type=['pdf'])


if uploaded_file is not None:
    file_path = os.path.join("/Users/nandan/Desktop/newRag/LightRAG/uploaded_doc/", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Use the temporary file path for further processing
    st.write(f"File saved temporarily at: {file_path}")

    loader = PyPDFLoader(file_path)
    document = loader.load()
    chunks = chunking_document(document)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    # Save FAISS index
    faiss.write_index(db.index, f"index/faiss_index_{uploaded_file.name}.index")

    # Save the document metadata separately, for example, the documents associated with the embeddings.
    with open(f"metadata/faiss_metadata_{uploaded_file.name}.pkl", "wb") as f:
        pickle.dump((db.docstore._dict, db.index_to_docstore_id), f)
    
    retriever = db.as_retriever()
    st.write("Document successfully uploaded!")
    st.text_area("Document Content", document[0].page_content)  # Just show the first 1000 characters

    rag2 = light_rag(file_path)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
#initialize session state for storing the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", key="user_input")
if user_input:
    # Generate answers using both RAG architectures
    answer1 = get_response(user_input, retriever, llm)
    answer2 = rag2.query(user_input, param=QueryParam(mode="global"))

    # Save bot responses to session state
    st.session_state.messages.append({"role": "bot_rag1", "content": answer1})
    st.session_state.messages.append({"role": "bot_rag2", "content": answer2})

    # Display answers side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RAG 1 Response")
        st.write(answer1)
    with col2:
        st.subheader("Light RAG Response")
        st.write(answer2)

# Conversation display
st.write("### Conversation History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    elif message["role"] == "bot_rag1":
        st.write(f"**RAG 1 Bot:** {message['content']}")
    elif message["role"] == "bot_Lightrag":
        st.write(f"**Light RAG  Bot:** {message['content']}")
