import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    print(f"Number of chunks created: {len(chunks)}")

    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

def get_vectorstore_local(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    
    retriever = vectorStore.as_retriever()
    
    template = """Answer the following question based only on the provided context:

    Context: {context}

    Question: {question}

    If you don't know the answer based on the context, just say "I don't have enough information to answer that."
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation.invoke(user_question)
        # Append user question and bot response to chat history
        st.session_state.chat_history.append((user_question, response))
    else:
        st.warning("Please upload and process PDF documents before asking questions.")

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple PDFs!", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.retriever = None
        st.session_state.chat_history = []  # Initialize chat history

    st.header("Chat with Multiple PDFs! :books:")
    user_question = st.text_input("Ask a Question about your documents:")

    if user_question:
        handle_userinput(user_question)

    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing!"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorStore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorStore)
                    st.session_state.retriever = vectorStore.as_retriever()
                st.success("Processing complete! You can now ask questions about your documents.")
            else:
                st.warning("Please upload PDF documents before processing.")

if __name__ == '__main__':
    main()
