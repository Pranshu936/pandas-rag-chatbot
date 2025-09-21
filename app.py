import os
import requests
import json
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SET UP API URL AND TOKEN ---
# This URL is based on your Hugging Face model repository
API_URL = "https://api-inference.huggingface.co/models/redtoxic/pandas-mistral-7b-finetuned" 


# IMPORTANT: Paste your Hugging Face Access Token here
# For better security, use an environment variable in a real project
HF_TOKEN = "YOUR_HF_ACCESS_TOKEN_HERE" 


# --- 2. LOAD THE LOCAL VECTOR DATABASE ---
print("🚀 Loading vector database...")
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
print("✅ Vector database loaded.")


# --- 3. DEFINE THE API-BASED INFERENCE FUNCTION ---
def answer_question(question):
    print(f"🔍 Received question: {question}")

    # Step 1: Retrieve relevant context from the local vector database
    print("... Retrieving context from FAISS...")
    retrieved_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 2: Format the prompt
    prompt = f"""<s>[INST] As a technical expert, please answer the question based on the following context.

### Context:
{context}

### Question:
{question} [/INST]"""

    # Step 3: Call the hosted model API
    print("... Calling hosted model API...")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.7, "return_full_text": False}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        return f"Error from API: {response.text}"

    result = response.json()
    answer = result[0].get('generated_text', "Sorry, I couldn't generate an answer.").strip()

    print(f"💬 Generated Answer: {answer}")
    return answer


# --- 4. CREATE AND LAUNCH THE GRADIO INTERFACE ---
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about the pandas library..."),
    outputs="markdown",
    title="🐼 Pandas Documentation Chatbot (API Version)",
    description="This chatbot is fine-tuned on pandas docs and served via a Hugging Face API.",
    allow_flagging="never"
)

print("\n🚀 Launching Gradio Interface...")
iface.launch()