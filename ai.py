# import streamlit as st
# import pandas as pd
# import numpy as np
# import ollama
# import PyPDF2

# # 1. Set page config MUST be the first Streamlit command
# st.set_page_config(page_title="T.A.S AI", page_icon="🤖")

# # 2. Inject custom CSS for the purple gradient background and white text
# custom_css = """
# <style>
# /* Target the main application container */
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(135deg, #f03830 0%, #7630f0 100%);
#     color: white;
# }

# /* Make the top header area transparent */
# [data-testid="stHeader"] {
#     background-color: transparent;
# }

# /* Force standard text elements to be white */
# h1, h2, h3, h4, h5, h6, p, span, div, label {
#     color: white !important;
# }

# /* Slightly darken the chat input box so it stands out */
# [data-testid="stChatInput"] {
#     background-color: rgba(0, 0, 0, 0.2) !important;
#     border-color: rgba(255, 255, 255, 0.3) !important;
# }
# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

# # 3. Application Header
# st.title("T.A.S AI 🖥️")
# st.markdown("##### Your friendly neighborhood AI personal assistant")

# MODEL_NAME = "qwen2.5"

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous chat messages from history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         # If the message contains our secret PDF wrapper tags, render it neatly
#         if "<PDF_CONTENT>" in message["content"]:
#             parts = message["content"].split("</PDF_CONTENT>\n\n")
#             st.markdown(parts[1]) # Show the user's typed question
#             with st.expander("📄 View attached PDF text"):
#                 st.markdown(parts[0].replace("<PDF_CONTENT>\n", ""))
#         else:
#             st.markdown(message["content"])

# # 4. Chat Input WITH built-in file uploader 
# if prompt := st.chat_input("Ask me anything...", accept_file=True, file_type=["pdf"]):
    
#     user_text = prompt.text
#     attached_files = prompt.files
#     final_prompt_to_save = ""

#     # CASE A: The user attached a PDF
#     if attached_files:
#         uploaded_pdf = attached_files[0]
#         with st.spinner(f"Reading {uploaded_pdf.name}..."):
#             try:
#                 pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
#                 extracted_text = ""
#                 for page in pdf_reader.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         extracted_text += page_text + "\n"
                
#                 # If they attached a file but didn't type a question, give the AI a default prompt
#                 query = user_text if user_text else f"Please analyze and summarize this attached document: {uploaded_pdf.name}"
                
#                 # Wrap the PDF in XML tags (LLMs read XML very well; Streamlit uses it to hide the text dump)
#                 final_prompt_to_save = f"<PDF_CONTENT>\n**Attached File:** `{uploaded_pdf.name}`\n\n{extracted_text}\n</PDF_CONTENT>\n\n{query}"

#             except Exception as e:
#                 st.error(f"Failed to read PDF: {e}")
#                 final_prompt_to_save = user_text
    
#     # CASE B: Standard text message (no PDF attached)
#     else:
#         final_prompt_to_save = user_text

#     # 5. Process and trigger AI response
#     if final_prompt_to_save:
#         st.session_state.messages.append({"role": "user", "content": final_prompt_to_save})

#         # Display the user's prompt instantly 
#         with st.chat_message("user"):
#             if "<PDF_CONTENT>" in final_prompt_to_save:
#                 parts = final_prompt_to_save.split("</PDF_CONTENT>\n\n")
#                 st.markdown(parts[1])
#                 with st.expander("📄 View attached PDF text"):
#                     st.markdown(parts[0].replace("<PDF_CONTENT>\n", ""))
#             else:
#                 st.markdown(final_prompt_to_save)

#         # Generate AI response
#         with st.chat_message("assistant"):
#             response_placeholder = st.empty()
#             full_response = ""
            
#             try:
#                 stream = ollama.chat(
#                     model=MODEL_NAME,
#                     messages=st.session_state.messages,
#                     stream=True,
#                 )
                
#                 for chunk in stream:
#                     full_response += chunk['message']['content']
#                     response_placeholder.markdown(full_response + "▌")
                
#                 response_placeholder.markdown(full_response)
#                 st.session_state.messages.append({"role": "assistant", "content": full_response})
                
#             except Exception as e:
#                 st.error(f"Failed to connect to Ollama: {e}")
#                 st.info("Make sure the Ollama application is running in the background.")

import streamlit as st
import pandas as pd
import numpy as np
import ollama
import PyPDF2  

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

# 3. Application UI Header
st.title("T.A.S AI 🖥️")
st.markdown("##### Your friendly neighborhood AI personal assistant")

MODEL_NAME = "qwen2.5"

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history on app rerun
for message in st.session_state.messages:
    # We hide the massive background PDF text dumps from the screen to keep the UI clean
    if not str(message["content"]).startswith("--- SYSTEM/BACKGROUND ---"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input WITH attachments enabled right in the chat bar
if prompt := st.chat_input("Ask me anything or attach a PDF...", accept_file=True):
    
    # 1. Check if the user uploaded any files
    if hasattr(prompt, "files") and prompt.files:
        for uploaded_file in prompt.files:
            if uploaded_file.name.lower().endswith('.pdf'):
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    try:
                        # Extract the text using PyPDF2
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text()])
                        
                        # Save the document text as a hidden background message for the AI to read
                        hidden_msg = f"--- SYSTEM/BACKGROUND ---\nThe user uploaded a document named '{uploaded_file.name}'. Here is the text content:\n\n{pdf_text}"
                        st.session_state.messages.append({"role": "user", "content": hidden_msg})
                        
                        # Show a quick pop-up notification instead of cluttering the chat
                        st.toast(f"Successfully processed {uploaded_file.name}!")
                        
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
            else:
                st.warning(f"The file {uploaded_file.name} was ignored. Currently, only PDFs are supported.")

    # 2. Extract the text the user typed
    user_text = prompt.text.strip() if hasattr(prompt, "text") and prompt.text else ""
    
    # If the user ONLY uploaded a file but didn't type a message, give the AI a default prompt
    if not user_text and hasattr(prompt, "files") and prompt.files:
        user_text = "I just uploaded a document. Please acknowledge it and provide a brief 2-3 sentence summary of what it is about."

    # 3. Process the chat text and get the AI response
    if user_text:
        # Display and save user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Generate and stream AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # NEW: Show the loading spinner while waiting for the AI to wake up and send its first word
                with st.spinner("T.A.S is thinking..."):
                    stream = ollama.chat(
                        model=MODEL_NAME,
                        messages=st.session_state.messages,
                        stream=True,
                    )
                    
                    # Manually pull the very first piece of text from the stream.
                    # Once this happens, the 'with st.spinner' block ends and the animation disappears!
                    try:
                        first_chunk = next(stream)
                        full_response += first_chunk['message']['content']
                    except StopIteration:
                        pass
                
                # Now that the spinner is gone, stream the rest of the response to the UI
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