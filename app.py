# app.py
import streamlit as st
from bs4 import BeautifulSoup
import requests
import pdfplumber
from docx import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

@st.cache_resource
def load_data():
    # --- Web Scraping ---
    url = 'https://www.angelone.in/support'
    response = requests.get(url)
    data_string = ""
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        data_string = soup.get_text(separator=' ', strip=True)

    # --- PDF Upload ---
    pdf_text = ""
    for uploaded_file in st.file_uploader("Upload up to 4 PDFs", type="pdf", accept_multiple_files=True):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text() + "\n"

    # --- DOCX Upload ---
    doc_text = ""
    docx_file = st.file_uploader("Upload DOCX file", type="docx")
    if docx_file:
        doc = Document(docx_file)
        doc_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

    full_text = data_string + pdf_text + doc_text

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.create_documents([full_text])

    # --- Embedding + FAISS ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # --- FLAN-T5 ---
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    # --- QA Chain ---
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                           return_source_documents=True)
    return qa_chain

qa_chain = load_data()

def ask_question(query):
    result = qa_chain(query)
    sources = result.get("source_documents", [])
    if not sources or len(" ".join([s.page_content for s in sources]).strip()) < 10:
        return "I don't know."
    return result["result"]

# --- Streamlit UI ---
st.title("RAG Chatbot with Angel One + Insurance Docs")
question = st.text_input("Ask a question:")
if question:
    with st.spinner("Searching..."):
        answer = ask_question(question)
    st.markdown("**Answer:** " + answer)
