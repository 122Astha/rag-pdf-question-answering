import streamlit as st
from rag_pipeline import load_and_split_pdf, create_vector_store, ask_question

st.title("📄 AI PDF Question Answering System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    chunks = load_and_split_pdf("temp.pdf")
    vector_store = create_vector_store(chunks)

    question = st.text_input("Ask a question about the PDF")

    if question:
        answer = ask_question(vector_store, question)
        st.write("### Answer:")
        st.write(answer)