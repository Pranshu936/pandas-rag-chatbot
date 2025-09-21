
# RAG Chatbot for Pandas Documentation with Fine-Tuned Mistral-7B

## Overview

This repository contains the source code for an end-to-end Retrieval-Augmented Generation (RAG) system that provides a question-answering interface for the official Pandas library documentation. The system leverages a FAISS vector database for efficient semantic retrieval and a `Mistral-7B-Instruct-v0.2` model fine-tuned using QLoRA for accurate, context-aware answer generation.

## System Architecture

The project follows a standard RAG pipeline:

1.  **Data Ingestion:** A Python script scrapes the official Pandas documentation, cleaning and extracting raw text content.
2.  **Data Processing:** The raw text is chunked into smaller, semantically meaningful segments suitable for embedding.
3.  **Indexing:** The text chunks are converted into vector embeddings using a `sentence-transformers` model and stored in a local FAISS vector index for efficient similarity search.
4.  **Fine-Tuning:** The `Mistral-7B-Instruct-v0.2` model is fine-tuned on an instruction dataset generated from the documentation chunks. This step uses QLoRA for memory-efficient training on a single GPU.
5.  **Retrieval & Generation:**
    * A user's query is converted into an embedding.
    * The FAISS index is queried to retrieve the most relevant document chunks (the "context").
    * The query and context are passed to the fine-tuned LLM via a structured prompt.
    * The LLM generates a final answer based on the provided context.
6.  **User Interface:** The entire pipeline is exposed through an interactive web UI built with Gradio.

## Tech Stack

| Area                  | Technologies                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| **LLM & Fine-Tuning** | `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `torch`                                       |
| **Retrieval & NLP** | `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `sentence-transformers`                       |
| **Data Pipeline** | `requests`, `beautifulsoup4`                                                                             |
| **Application UI** | `gradio`                                                                                                 |

## Repository Structure

```

.
├── faiss\_index/                  \# Generated FAISS vector store
├── pandas-mistral-7b-finetuned/  \# Fine-tuned LoRA adapters from Colab
├── .gitignore                    \# Git ignore file
├── README.md                     \# This README file
├── app.py                        \# Main Gradio application for local inference
├── create\_vector\_db.py           \# Script to create the FAISS index
├── process\_data.py               \# Script to clean and chunk scraped data
├── requirements.txt              \# Project dependencies
└── scraper.py                    \# Script to scrape Pandas documentation

````

## Setup and Installation

**Prerequisites:**
* Python 3.10+
* Git
* An NVIDIA GPU with >= 16 GB VRAM for local inference.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Pranshu936/pandas-rag-chatbot.git](https://github.com/Pranshu936/pandas-rag-chatbot.git)
    cd pandas-rag-chatbot
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate.ps1
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Execution Workflow

Execute the scripts in the following order to build and run the application.

### Step 1: Data Ingestion & Processing

These scripts create the knowledge base for the RAG system.

1.  **Scrape Documentation:**
    ```bash
    python scraper.py
    ```
    * **Input:** None
    * **Output:** `scraped_pandas_docs.json`

2.  **Process and Chunk Data:**
    ```bash
    python process_data.py
    ```
    * **Input:** `scraped_pandas_docs.json`
    * **Output:** `processed_chunks.json`

### Step 2: Vector Database Creation

This script embeds the processed text chunks and builds the FAISS index.

```bash
python create_vector_db.py
````

  * **Input:** `processed_chunks.json`
  * **Output:** `faiss_index/` directory

### Step 3: Model Fine-Tuning (Google Colab)

This step requires a GPU and should be performed in a cloud environment like Google Colab.

1.  **Environment:** Open a new notebook in Google Colab and set the runtime to **T4 GPU**.
2.  **Upload Data:** Upload the `processed_chunks.json` file to your Colab session.
3.  **Execute Fine-Tuning:** Run the commands from the `Fine_Tuning_Notebook.ipynb` provided in the repository. This will fine-tune the model and save the adapters.
4.  **Download Artifacts:** Download the resulting `pandas-mistral-7b-finetuned` folder and place it in the root of your local project directory.

### Step 4: Local Inference & UI

This script loads all the components and launches the interactive user interface.

**Prerequisites:**

  * The `faiss_index/` directory must exist.
  * The `pandas-mistral-7b-finetuned/` directory must be present.

**Run the application:**

```bash
python app.py
```

The script will download the base Mistral-7B model (\~15 GB) on the first run. It will then load the model and vector database into memory and provide a local URL (e.g., `http://127.0.0.1:7860`) for the Gradio UI.
