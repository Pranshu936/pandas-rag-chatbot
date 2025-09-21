import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Idhar hum FAISS vector database create karne wale hain
def create_vector_database(input_filename='processed_chunks.json', index_path='faiss_index'):
    """
    Processed text chunks se FAISS vector database banayega
    """
    print("🚀 Starting vector database creation...")

    # 1. Load processed text chunks
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' nahi mila. Pehle process_data.py chala lo bhai.")
        return

    # List of dict ko LangChain Document objects mein convert kar rahe hain
    # Kyunki FAISS vector store ko yahi format chahiye
    documents = [
        Document(page_content=chunk['text'], metadata={'source': chunk['source']})
        for chunk in chunks_data
    ]
    print(f"📄 Loaded {len(documents)} document chunks.")

    # 2. Initialize embedding model
    # Yeh model text ko numerical vectors mein convert karega
    print("🧠 Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # 3. Create the FAISS vector store
    # Idhar model download hoga, embeddings create honge, aur index banega
    # Pehli baar thoda time lag sakta hai
    print("💻 Creating FAISS vector store. Thoda wait karo bhai...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("✅ Vector store ready ho gaya.")

    # 4. Save the vector store locally
    vector_store.save_local(index_path)
    print(f"\n✅ Success! Vector store save ho gaya idhar -> '{index_path}'")


if __name__ == "__main__":
    create_vector_database()
