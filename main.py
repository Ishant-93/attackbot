import PyPDF2
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time
import os
from dotenv import load_dotenv
# Set your OpenAI API key
load_dotenv()

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

# Function to build LangChain LLM for extracting insights
def create_llm_chain(template): 
    prompt = PromptTemplate.from_template(template) 
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  
    chain = prompt | llm  
    return chain

# Streamlit UI setup
st.set_page_config(page_title="Attacked.ai", layout="wide")

# Inject custom CSS to make the chat input smaller
st.markdown("""
    <style>
    .stTextInput {
        height: 30px;  /* Adjust height as needed */
        font-size: 14px;  /* Adjust font size as needed */
    }
    </style>
    """, unsafe_allow_html=True)

# Display the logo with centered alignment
logo_file = 'main_logo.png'
if logo_file:
    st.image(logo_file, width=200)

# App Title
st.title("Attacked.Ai")

# Initialize session state for chat history and PDF text
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = None

# Sidebar for PDF upload
st.sidebar.header("Upload PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file (optional)", type="pdf", key="pdf_uploader")

if pdf_file:
    pdf_text = extract_text_from_pdf(pdf_file)
    st.session_state["pdf_text"] = pdf_text

# Button to unload the PDF
if st.sidebar.button("Unload PDF"):
    st.session_state["pdf_text"] = None
    st.sidebar.success("PDF unloaded successfully.")

# Display chat history
for chat in st.session_state["chat_history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Handle user input for chat
if prompt := st.chat_input("What is up?"):
    # Append user input to chat history
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the prompt with chat history
    history_prompt = "\n".join([f"{chat['role']}: {chat['content']}" for chat in st.session_state["chat_history"]])
    
    # Create response based on PDF content if available
    template = ""
    if st.session_state["pdf_text"]:
        template = """You are Attacked.ai, an advanced simulation platform that tests the security posture of organizations by simulating various attack types (e.g., ransomware, DDoS, supply chain attacks) in real-time. The platform uses AI to create scenarios that reflect real-world vulnerabilities and provides users with insights into defense mechanisms and mitigation strategies. You are a helpful assistant that provides detailed answers.

        [PDF text] : {text}
        If the user has provided a PDF, use the information from the PDF text to enhance your response. Specifically, look for details in the report that may directly address the user's query or offer additional context. Summarize relevant sections and integrate them into your response to ensure accurate and helpful insights. If the PDF is a financial report, prioritize extracting key figures, trends, and analysis.

        [HISTORY]: {history}

        [User]: {user_input}

        Assistant:
        """

    else:
        template = """[System Information] : You are Attacked.ai which is an advanced simulation platform that tests the security posture of organizations by simulating various attack types (e.g., ransomware, DDoS, supply chain attacks) in real-time. The platform uses AI to create scenarios that reflect real-world vulnerabilities and provide users with insights into defense mechanisms and mitigation strategies. You are a helpful assistant that provides detailed answers.
        [HISTORY]: {history}
        
        [User]: {user_input}
        
        [Formatting Guidelines]
        Respond with clear and concise information, using bullet points where appropriate.

        Assistant:
        """
    
    message_placeholder = st.empty()
    chain = create_llm_chain(template=template)

    # Generate response with chat history and PDF content if applicable
    inputs = {"history": history_prompt, "user_input": prompt}
    if st.session_state["pdf_text"]:
        inputs["text"] = st.session_state["pdf_text"]
    
    # Get the response and stream it
    response = chain.invoke(inputs)
    if response and hasattr(response, 'content'):
        full_response = response.content

        # Stream the response with a cursor
        full = ""
        for chunk in full_response:
            full += chunk
            message_placeholder.markdown(full + '|', unsafe_allow_html=True)  # Show streaming response with a cursor
            time.sleep(0.01)  # Control the speed of the streaming

        # Final display without the cursor
        message_placeholder.markdown(full, unsafe_allow_html=True)
    else:
        message_placeholder.markdown("No response received.")

    # Append assistant's response to the chat history
    st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
