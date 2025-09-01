import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# load groq api key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    
    Answer the question based on the context provided. If the context does not contain the answer, say "I don't know".
    <context>
    {context}
    <context>
    Question: {input}
    
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("D:\\LangChain Projects\\RAG DOCUMENT Q&A\\research_papers")
        st.session_state.documents = st.session_state.loader.load()

        st.write(f"📄 Loaded {len(st.session_state.documents)} PDF documents.")

        if not st.session_state.documents:
            st.error("❌ No PDF documents found! Check folder or PDF format.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])

        st.write(f"✂️ Split into {len(st.session_state.final_documents)} document chunks.")

        if not st.session_state.final_documents:
            st.error("❌ No document chunks created! Check if PDFs contain text.")
            return

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.success("✅ FAISS index created successfully.")

        
st.title("RAG Document Q&A")        
user_prompt = st.text_input("Enter your question here: ")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Document embeddings created successfully!")
    
import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time : {time.process_time() - start} seconds")
    
    st.write(response['answer'])
  
    # with a streamlit expander
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            