import streamlit as st
from utils import (
    extract_pdf_elements, generate_text_summaries, generate_image_summary,
    create_multivector_retriever, query_multimodal_rag, decode_base64_image
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import time

# Initialize Streamlit app
st.set_page_config(page_title="Multimodal RAG System", layout="wide")
st.title("📄 Multimodal RAG System")

# Sidebar for PDF Upload
st.sidebar.header("Upload Your Research Paper")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Input for user queries
query = st.text_input("🔍 Ask a Question About Your PDF:", placeholder="Type your query here...")

# Initialize variables
retriever = None

# Process PDF and Generate Summaries
if uploaded_file:
    st.sidebar.success("✅ PDF Uploaded Successfully!")
    st.sidebar.info("⏳ Extracting content...")

    # Extract elements from the uploaded PDF
    start_time = time.time()
    text_data, table_data, image_data = extract_pdf_elements(uploaded_file)

    # Generate Summaries
    text_summaries, table_summaries = generate_text_summaries(text_data, table_data, summarize_texts=True)
    image_summaries = generate_image_summary(image_data)

    # Store Data in ChromaDB
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_storage"
    )
    retriever = create_multivector_retriever(vectorstore, 
                                             text_summaries, text_data, 
                                             table_summaries, table_data, 
                                             image_summaries, image_data)

    st.success("✅ Extraction and Storage Complete!")
    st.sidebar.success(f"Processing Time: {round(time.time() - start_time, 2)} seconds")

# Handle User Queries
if st.button("Submit Query"):
    if uploaded_file is None:
        st.warning("⚠ Please upload a PDF first.")
    elif query.strip() == "":
        st.warning("⚠ Please enter a valid query.")
    elif retriever is None:
        st.error("❌ Error: Document not processed yet. Please upload and process the PDF.")
    else:
        st.info("⏳ Retrieving answer...")

        # Run the multimodal RAG pipeline
        response = query_multimodal_rag(retriever, query)
        retrieved_context = response.get("context", {})
        answer = response.get("response", "No answer generated.")

        # Display Answer
        st.subheader("💡 Generated Answer")
        st.write(answer)

        # Display Retrieved Content
        st.subheader("📋 Retrieved Context")
        if retrieved_context.get("texts"):
            st.markdown("**Relevant Text:**")
            for text_chunk in retrieved_context["texts"]:
                st.write(text_chunk)

        if retrieved_context.get("images"):
            st.markdown("**Relevant Images:**")
            for img_base64 in retrieved_context["images"]:
                st.image(decode_base64_image(img_base64), caption="Relevant Image", use_column_width=True)

        st.success("✅ Query Processed Successfully!")
