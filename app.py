import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="Document QA Bot")
st.title("📄 Document Question Answering Bot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector store (Chroma instead of FAISS)
    db = Chroma.from_documents(chunks, embeddings)

    st.success("✅ Document processed! Ask your question below 👇")

    query = st.text_input("Ask a question")

    if query:
        # Retrieve similar docs
        retrieved_docs = db.similarity_search(query)

        # Load local HuggingFace model
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # QA Chain
        chain = load_qa_chain(llm, chain_type="stuff")

        result = chain.invoke({
            "input_documents": retrieved_docs,
            "question": query
        })

        st.write("### 📌 Answer:")
        st.write(result["output_text"])
