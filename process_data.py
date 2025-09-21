import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
    """
    Cleans the input text by removing extra whitespace, code syntax, and other noise.
    """
    # Remove code block syntax like In [...]: and Out[...]:
    text = re.sub(r'In \[\d+\]: |Out\[\d+\]:', '', text)
    
    # Remove common page navigation and footer text
    text = re.sub(r'previous\s*Enhancing performance\s*next.*', '', text, flags=re.DOTALL)
    text = re.sub(r'On this page.*', '', text, flags=re.DOTALL)
    
    # Replace multiple newlines and spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and section markers like # or ¶
    text = re.sub(r'#|¶', '', text)
    
    return text.strip()

def process_and_chunk_data(input_filename='scraped_pandas_docs.json', output_filename='processed_chunks.json'):
    """
    Loads scraped data, cleans it, chunks it, and saves the result.
    """
    print("🚀 Starting data processing and chunking...")

    # 1. Load the scraped data
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Please run the scraper first.")
        return

    print(f"📄 Loaded {len(scraped_data)} documents.")

    all_chunks = []
    
    # 2. Initialize the text splitter
    # This splitter tries to keep paragraphs/sentences together, which is good for context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max size of each chunk
        chunk_overlap=200   # Overlap between chunks to maintain context
    )

    # 3. Process each document
    for doc in scraped_data:
        if not doc.get('text'):
            continue
            
        # Clean the text
        cleaned_text = clean_text(doc['text'])
        
        # Split the text into chunks
        chunks = text_splitter.split_text(cleaned_text)
        
        # Add chunks with metadata to our list
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                'source': doc['url'],
                'text': chunk_text,
                'chunk_id': f"{doc['url']}#{i}" # Unique ID for each chunk
            })

    print(f"🧩 Created {len(all_chunks)} chunks.")

    # 4. Save the processed chunks to a new JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Success! Processed data saved to '{output_filename}'")

if __name__ == "__main__":
    process_and_chunk_data()