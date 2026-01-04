import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. SETUP & KEYS
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. LOAD YOUR 11 RULE FILES
print("AI DUDE is reading your Mitsubishi ST Rules...")
loader = DirectoryLoader(
    './docs/st_rules/', 
    glob="./*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

# 3. SPLIT RULES INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 4. CREATE THE VECTOR DATABASE (The Library Catalog)
# This uses a free local model to 'understand' your text
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("Knowledge Base Ready!")

# 5. SYSTEM PROMPT
with open("prompts/system_prompt.txt", "r") as f:
    system_instruction = f.read()

# 6. CHAT LOOP
while True:
    user_query = input("\nUSER: ")
    if user_query.lower() in ["quit", "exit"]: break

    # RAG: Find the most relevant rules from your .txt files
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Combine Rules + System Prompt + Question
    full_prompt = f"""
    SYSTEM RULES: {system_instruction}
    
    REFERENCE RULES FROM MANUALS:
    {context}
    
    USER QUESTION: {user_query}
    """

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": full_prompt}]
        )
        print(f"\nAI DUDE: {chat_completion.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")