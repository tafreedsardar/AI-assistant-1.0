import streamlit as st
import pandas as pd
import numpy as np

st.title('AI model that actually works')


import ollama

# Set up the page title
st.set_page_config(page_title="Local Ollama Chat", page_icon="🤖")
st.title("AI Chatbot")


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
