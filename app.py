import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template
import os
import requests

os.environ['CURL_CA_BUNDLE'] = ''
requests.packages.urllib3.disable_warnings()

def create_vectorstore(text_chunks):
    """Create a vector store from text chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings() 
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def extract_text_from_pdfs(pdf_docs):
    """Extract text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def split_text_into_chunks(text):
    """Split text into chunks for the vector store."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


def create_vectorstore(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = ChatGroq()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def create_conversation_chain(vectorstore):
    """Create a conversational retrieval chain."""
    llm = ChatGroq()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def display_chat_history():
    """Display the chat history in the Streamlit app."""
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def handle_user_input(user_question):
    """Handle the user's question and update the chat history."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    display_chat_history()


def setup_streamlit_app():
    """Setup the Streamlit app layout and state."""
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def create_vectorstore(text_chunks):
    """Create a vector store from text chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings() 
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def main():
    setup_streamlit_app()
    st.header("Chat with multiple PDFs :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdfs(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                vectorstore = create_vectorstore(text_chunks)
                st.session_state.conversation = create_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()