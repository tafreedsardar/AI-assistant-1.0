import streamlit as st
import pandas as pd
import numpy as np
import ollama
import PyPDF2  

# first Streamlit command
st.set_page_config(page_title="T.A.S AI", page_icon="🤖")

# Inject custom CSS
custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f03830 0%, #7630f0 100%);
    color: white;
}

/* top header area transparent */
[data-testid="stHeader"] {
    background-color: transparent;
}

/* white text */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: white !important;
}

/* darken the chat input box */
[data-testid="stChatInput"] {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Application UI
st.title("T.A.S AI 🖥️")
st.markdown("##### Your friendly neighborhood AI personal assistant")

MODEL_NAME = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"

# Initialize session states
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = []

# Helper function to split text into manageable overlapping chunks
def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Display previous chat messages (filtering out system/background logs)
for message in st.session_state.messages:
    if not str(message["content"]).startswith("--- SYSTEM/BACKGROUND ---") and not message.get("is_hidden", False):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything or attach a PDF...", accept_file=True):
    
    # Check if the user uploaded any files
    if hasattr(prompt, "files") and prompt.files:
        for uploaded_file in prompt.files:
            if uploaded_file.name.lower().endswith('.pdf'):
                with st.spinner(f"Extracting & Indexing text from {uploaded_file.name}..."):
                    try:
                        # Extract the text
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text()])
                        
                        # --- CHUNKING & RAG INDEXING ---
                        file_chunks = chunk_text(pdf_text)
                        
                        # Generate vector embeddings via Ollama
                        file_embeddings = []
                        for chunk in file_chunks:
                            response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                            file_embeddings.append(response["embedding"])
                        
                        st.session_state.pdf_chunks.extend(file_chunks)
                        st.session_state.pdf_embeddings.extend(file_embeddings)
                        
                        st.toast(f"Successfully processed and indexed {uploaded_file.name}!")
                        
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
            else:
                st.warning(f"The file {uploaded_file.name} was ignored. Currently, only PDFs are supported.")

    user_text = prompt.text.strip() if hasattr(prompt, "text") and prompt.text else ""
    
    if not user_text and hasattr(prompt, "files") and prompt.files:
        user_text = "I just uploaded a document. Summarize what it is about based on its contents."

    if user_text:
        # Save user message to history
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # --- RETRIEVAL PHASE ---
        context_str = ""
        if st.session_state.pdf_embeddings and st.session_state.pdf_chunks:
            with st.spinner("Searching document data..."):
                try:
                    # Get user question vector embedding
                    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=user_text)["embedding"]
                    
                    # Compute Cosine Similarity via Dot Product
                    scores = np.dot(st.session_state.pdf_embeddings, query_embedding)
                    
                    # Grab top 3 closest matching chunks
                    top_indices = np.argsort(scores)[::-1][:3]
                    retrieved_chunks = [st.session_state.pdf_chunks[idx] for idx in top_indices]
                    context_str = "\n\n".join(retrieved_chunks)
                except Exception as e:
                    st.warning(f"Failed semantic retrieval step: {e}")

        # Assemble the payload array for Ollama
        working_messages = []
        
        # Insert background context if relevant text pieces were found
        if context_str:
            working_messages.append({
                "role": "user", 
                "content": f"--- SYSTEM/BACKGROUND ---\nContext information extracted from document sources:\n{context_str}\n\nUse only the context above to respond to the user query below if applicable."
            })
        
        # Append existing conversational threads
        working_messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

        # Generate AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner("T.A.S is thinking..."):
                    stream = ollama.chat(
                        model=MODEL_NAME,
                        messages=working_messages,
                        stream=True,
                    )
                    
                    try:
                        first_chunk = next(stream)
                        full_response += first_chunk['message']['content']
                    except StopIteration:
                        pass
                
                response_placeholder.markdown(full_response + "▌")
                for chunk in stream:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Failed to connect: {e}")
                st.info("Oops there is an error.")