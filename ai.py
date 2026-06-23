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
/* Target the main application container */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f03830 0%, #7630f0 100%);
    color: white;
}

/* Make the top header area transparent */
[data-testid="stHeader"] {
    background-color: transparent;
}

/* Force standard text elements to be white */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: white !important;
}

/* Slightly darken the chat input box so it stands out, but keep text white */
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

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history on app rerun
for message in st.session_state.messages:
    if not str(message["content"]).startswith("--- SYSTEM/BACKGROUND ---"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything or attach a PDF...", accept_file=True):
    
    # Check if the user uploaded any files
    if hasattr(prompt, "files") and prompt.files:
        for uploaded_file in prompt.files:
            if uploaded_file.name.lower().endswith('.pdf'):
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    try:
                        # Extract the text using PyPDF2
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text()])
                        
                        hidden_msg = f"--- SYSTEM/BACKGROUND ---\nThe user uploaded a document named '{uploaded_file.name}'. Here is the text content:\n\n{pdf_text}"
                        st.session_state.messages.append({"role": "user", "content": hidden_msg})
                        
                        st.toast(f"Successfully processed {uploaded_file.name}!")
                        
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
            else:
                st.warning(f"The file {uploaded_file.name} was ignored. Currently, only PDFs are supported.")

    user_text = prompt.text.strip() if hasattr(prompt, "text") and prompt.text else ""
    
    # If the user ONLY uploaded a file but didn't type a message, give the AI a default prompt
    if not user_text and hasattr(prompt, "files") and prompt.files:
        user_text = "I just uploaded a document. Please acknowledge it and provide a brief 2-3 sentence summary of what it is about."

    if user_text:
        # Display and save user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Generate AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
        # Spinner to indicate that the AI is processing the request            
            try:
                with st.spinner("T.A.S is thinking..."):
                    stream = ollama.chat(
                        model=MODEL_NAME,
                        messages=st.session_state.messages,
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
                
                # Remove cursor indicator and show final text
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Failed to connect to Ollama: {e}")
                st.info("Make sure the Ollama desktop application or server is running.")