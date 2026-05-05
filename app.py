import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Document QA Bot")
st.title("📄 Document Question Answering Bot")

# 🔑 Groq API Key
client = Groq(api_key="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")

# ---------------- FILE UPLOAD ----------------
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

    st.success("✅ Document processed!")

    query = st.text_input("Ask a question")

    if query:
        # Simple retrieval
        context = " ".join([doc.page_content for doc in chunks[:3]])

        prompt = f"""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {query}
        """

        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

            st.write("### 📌 Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
