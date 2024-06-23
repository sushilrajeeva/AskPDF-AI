# Install and import streamlit to create UI
# Install and import pypdf2 to read our pdf
# Install and import langchain to intereact with our LLM
# Install and import faiss-cpu as our vector store
# Install and import openai and huggingface_hub to create create LLMs

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

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

                # Get the text chunks (dividing the pdf into multiple small chunks)

                # Create a vector store to store the embeddings

if __name__ == '__main__':
    main()