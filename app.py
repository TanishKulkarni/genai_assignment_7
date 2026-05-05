import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Document QA Bot")
st.title("📄 Document Question Answering Bot")

# 🔑 Add your Hugging Face API key here
API_KEY = "your_huggingface_api_key"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Function to query model
def query_hf(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

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

    st.success("✅ Document processed! Ask your question below 👇")

    query = st.text_input("Ask a question")

    if query:
        # Simple retrieval (top chunks)
        context = " ".join([doc.page_content for doc in chunks[:3]])

        prompt = f"""
        Answer the question based on the context below:

        Context:
        {context}

        Question:
        {query}
        """

        result = query_hf({"inputs": prompt})

        st.write("### 📌 Answer:")
        if isinstance(result, list):
            st.write(result[0]["generated_text"])
        else:
            st.write("Error:", result)
