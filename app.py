# Install and import streamlit to create UI
# Install and import pypdf2 to read our pdf
# Install and import langchain to intereact with our LLM
# Install and import faiss-cpu as our vector store
# Install and import openai and huggingface_hub to create create LLMs

import streamlit as st

def main():

    st.set_page_config(page_title="Chat with Multiple PDFs!", page_icon= ":books:")
    st.header("Chat with Multiple PDFs! :books:")
    st.text_input("Ask a Question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your PDFs here and click 'Process'")
        st.button("Process")

if __name__ == '__main__':
    main()