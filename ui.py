import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. UI CONFIGURATION (Catchy Title) ---
st.set_page_config(page_title="AI DUDE", page_icon="🤖")

st.markdown("<h3 style='text-align: center; color: #E60012;'>NEXUS AI PRESENTS</h3>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🤖 AI DUDE: Mitsubishi PLC Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><i>The Intelligent Guide for Structured Text & GX Works3</i></p>", unsafe_allow_html=True)
st.divider()

# --- 2. INITIALIZE AI & RAG (Cached for speed) ---
@st.cache_resource
def initialize_knowledge_base():
    load_dotenv()
    loader = DirectoryLoader('./docs/st_rules/', glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore, Groq(api_key=os.getenv("GROQ_API_KEY"))

vectorstore, client = initialize_knowledge_base()

# Load System Prompt
with open("prompts/system_prompt.txt", "r") as f:
    system_instruction = f.read()

# --- 3. CHAT HISTORY MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. THE CHAT INTERFACE ---
if prompt := st.chat_input("Ask about Mitsubishi PLC Rules..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    docs = vectorstore.similarity_search(prompt, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    full_prompt = f"SYSTEM RULES: {system_instruction}\n\nREFERENCE RULES: {context}\n\nUSER: {prompt}"

    # Generate AI Response
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": full_prompt}]
            )
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")