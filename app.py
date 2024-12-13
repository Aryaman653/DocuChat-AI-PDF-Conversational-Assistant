import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="All-MiniLM-L6-v2")

# Streamlit UI
st.title("DocuChat AI: Intelligent PDF Conversational Assistant")
st.write("Turn Your PDFs into Chat Partners!")
st.write("AI-Powered Q&A with PDF Insights and Chat Memory!")


# Input Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma-7b-It")

    session_id = st.text_input("Session ID", value="default")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            # Load PDF content
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            if docs:
                documents.extend(docs)
                st.write(f"Loaded {len(docs)} documents from {uploaded_file.name}")
            else:
                st.warning(f"Could not load any content from {uploaded_file.name}")

        if documents:
            # Embeddings and text splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            st.write(f"Split into {len(docs)} chunks.")

            db = FAISS.from_documents(documents=docs,embedding=embeddings)
            retriever = db.as_retriever()

            # Adding chat history
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Prompt Template
            system_prompt = (
                "You are a question-answering assistant. Use the provided retrieved context to respond to the user's query. If the answer is not available, acknowledge it clearly. Limit your response to three concise sentences."
                "{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            with_message_history = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = with_message_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.write("Assistant:")
                st.write(response["answer"])

        else:
            st.error("No valid content found in the uploaded PDFs.")
else:
    st.write("Please enter Groq API key.")
