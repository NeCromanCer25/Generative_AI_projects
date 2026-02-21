import os
import streamlit as st
from dotenv import load_dotenv

#langchain imports
from langchain_classic import text_splitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Page configuration
st.set_page_config(
    page_title="C++ Rag Chatbot",
    page_icon="🐼"
)
st.title("🍆 C++ Rag Chatbot")
st.write("Ask any question related to C++ introduction")

# Step 2: load env var
load_dotenv()

# Step 3: Cache document loading
@st.cache_resource

def load_vector_store():
    # 1-Load Documents
    loader = TextLoader("C++_Introduction.txt",encoding="utf-8")
    docx = loader.load()

    # 2-Split Text
    txtsplit = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_docx = txtsplit.split_documents(docx)

    # 3-Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name = "all-miniLM-L6-v2"
        # This is a embedding model
    )

    # 4-Create FAISS vector store
    db = FAISS.from_documents(final_docx,embeddings)
    return db

#user input
db = load_vector_store()
query = st.text_input("Enter your query about C++")
if query:
    dox = db.similarity_search(query, k=3)
    st.subheader("Retrieved context")
    for i , doc in enumerate(dox):
        st.markdown(f"**Result {i+1} : **")

        st.write(doc.page_content)
