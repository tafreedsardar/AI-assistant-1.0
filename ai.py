import streamlit as st
import pandas as pd
import numpy as np
import ollama

# 1. Set page config MUST be the first Streamlit command
st.set_page_config(page_title="T.A.S AI", page_icon="🤖")

# 2. Inject custom CSS for the purple gradient background and white text
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

# 3. Rest of your application UI
st.title("T.A.S AI 🖥️")
st.markdown("##### Your friendly neighborhood AI personal assistant")

MODEL_NAME = "qwen2.5"

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Ask me anything..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response placeholder
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Connect to Ollama API and request a streaming response
            stream = ollama.chat(
                model=MODEL_NAME,
                messages=st.session_state.messages,
                stream=True,
            )
            
            # Stream chunks into the UI as they arrive
            for chunk in stream:
                full_response += chunk['message']['content']
                response_placeholder.markdown(full_response + "▌")
            
            # Remove cursor indicator and show final text
            response_placeholder.markdown(full_response)
            
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {e}")
            st.info("Make sure the Ollama desktop application or server is running.")