# Install and import streamlit to create UI
# Install and import pypdf2 to read our pdf
# Install and import langchain to intereact with our LLM
# Install and import faiss-cpu as our vector store
# Install and import openai and huggingface_hub to create create LLMs

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

def get_pdf_text(pdf_docs):
    # This funciton takes pdf document lists and return a single string of text with all of the text content of the pdf
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # pdf reader creates an object of the pdf that has multiple pages 
        # each page has content of the pdf in it which we can extract using extract_text() method
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    """
    Takes raw text as input (string) and returns a list of chunks of text that can be fed into a vector database.
    Uses the CharacterTextSplitter from langchain to divide text into chunks/paragraphs.
    
    Args:
        raw_text (str): The raw text to be split into chunks.
        
    Returns:
        list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Corrected to use newline character
        chunk_size=1000,  # Chunk size of 1000 characters
        chunk_overlap=200,  # Overlap of 200 characters to handle incomplete chunks
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    print(f"Number of chunks created: {len(chunks)}")

    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vetorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vetorStore

def get_vectorstore_local(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vetorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vetorStore


def main():

    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple PDFs!", page_icon= ":books:")
    st.header("Chat with Multiple PDFs! :books:")
    st.text_input("Ask a Question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            # To show a spinner bar to show user the process is processing
            with st.spinner("Processing!"):

                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Get the text chunks (dividing the pdf into multiple small chunks)
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                

                # Create a vector store to store the embeddings
                # 1. using open ai embedding model (paid)
                # creating vector store
                # vectorStore = get_vectorstore(text_chunks)
                # print("vectorstore", vectorStore)
                # 2. using instructor-xl (free)
                vectorStore_local = get_vectorstore_local(text_chunks)
                print("vectorstore", vectorStore_local)

if __name__ == '__main__':
    main()