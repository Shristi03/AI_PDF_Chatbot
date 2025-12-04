import streamlit as st
import os
import shutil

# --- THE FIX IS HERE ---
# We changed 'from rag_project' to 'from Pdfload' to match your actual file.
from Pdfload import load_and_chunk_pdfs, setup_vector_db, query_rag_system, chroma_client

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat AI", page_icon="📚", layout="wide")

st.title("📚 Chat with your PDF Documents")

# --- SIDEBAR: MANAGE FILES ---
with st.sidebar:
    st.header("📂 Document Manager")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload new PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process & Ingest Files"):
            with st.spinner("Processing files..."):
                # 1. Save uploaded files to 'pdfs' folder
                if not os.path.exists("pdfs"):
                    os.makedirs("pdfs")

                for uploaded_file in uploaded_files:
                    # Save to the specific 'pdfs' folder
                    file_path = os.path.join("pdfs", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # 2. Trigger the ingestion pipeline from your Pdfload script
                # We pass the folder name "pdfs"
                docs, meta, ids = load_and_chunk_pdfs("pdfs")

                if docs:
                    # Rebuild the DB
                    setup_vector_db(docs, meta, ids)
                    st.success(f"✅ Successfully ingested {len(uploaded_files)} files!")
                else:
                    st.error("No valid text found in PDFs.")

    # Show existing files
    if os.path.exists("pdfs"):
        st.write("---")
        st.write("**Current Files in Knowledge Base:**")
        try:
            files = os.listdir("pdfs")
            if not files:
                st.caption("No files yet.")
            for f in files:
                st.caption(f"📄 {f}")
        except FileNotFoundError:
            st.caption("Folder not created yet.")

# --- CHAT INTERFACE ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the answer from your RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Connect to the DB (it's persistent, so we just load it)
                collection = chroma_client.get_collection(name="my_pdf_knowledge_base")

                # Call your existing function
                response = query_rag_system(collection, prompt)

                st.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})