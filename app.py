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
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from htmlTemplates import css, bot_template, user_template

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
        separator="\n", # Corrected to use newline character
        chunk_size=1000, # Chunk size of 1000 characters
        chunk_overlap=200, # Overlap of 200 characters to handle incomplete chunks
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
    
    system_template = """You are a helpful AI assistant. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        chat_history = st.session_state.chat_history
        
        response = st.session_state.conversation.invoke({
            "question": user_question, 
            "chat_history": chat_history
        })
        
        st.session_state.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=response)
        ])
    else:
        st.warning("Please upload and process PDF documents before asking questions.")

def main():
    load_dotenv()

    st.set_page_config(page_title="AskPDF-AI!", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.retriever = None
        # This makes the application when re run will see if the application is in conversation or not if not it will set to none

    st.header("Chat with Multiple PDFs! :books:")
    user_question = st.text_input("Ask a Question about your documents:")

    if user_question:
        handle_userinput(user_question)

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                # To show a spinner bar to show user the process is processing
                with st.spinner("Processing!"):

                    # Get the pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the text chunks (dividing the pdf into multiple small chunks)
                    text_chunks = get_text_chunks(raw_text)

                    # Create a vector store to store the embeddings
                    # 1. using open ai embedding model (paid)
                    # creating vector store
                    # vectorStore = get_vectorstore(text_chunks)
                    # print("vectorstore", vectorStore)
                    # 2. using instructor-xl (free)
                    vectorStore = get_vectorstore(text_chunks)

                    # Create Conversation Chain
                    st.session_state.conversation = get_conversation_chain(vectorStore)
                    st.session_state.retriever = vectorStore.as_retriever()
                st.success("Processing complete! You can now ask questions about your documents.")
            else:
                st.warning("Please upload PDF documents before processing.")

if __name__ == '__main__':
    main()
