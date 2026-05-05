import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Document QA Bot")
st.title("📄 Document Question Answering Bot")

# 🔑 Replace with your Hugging Face API Key
API_KEY = "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"

# Use smaller model for faster response
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# -------------------- FUNCTION --------------------
def query_hf(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # If API fails
        if response.status_code != 200:
            return {"error": f"Status {response.status_code}: {response.text}"}

        try:
            return response.json()
        except Exception:
            return {"error": "Invalid JSON response", "raw": response.text}

    except Exception as e:
        return {"error": str(e)}

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file
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

    # -------------------- QUERY --------------------
    query = st.text_input("Ask a question")

    if query:
        # Take top chunks (simple retrieval)
        context = " ".join([doc.page_content for doc in chunks[:3]])

        prompt = f"""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {query}
        """

        result = query_hf({"inputs": prompt})

        st.write("### 📌 Answer:")

        # -------------------- RESPONSE HANDLING --------------------
        if "error" in result:
            st.error(result["error"])

        elif isinstance(result, list):
            st.write(result[0].get("generated_text", "No output"))

        else:
            st.write(result)
