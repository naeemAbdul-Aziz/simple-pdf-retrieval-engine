# 📄 PDF Document Chunking and Embedding with LangChain + Chroma

This project demonstrates how to load a PDF, split it into manageable chunks, embed the text using HuggingFace embeddings, store the vectors using Chroma, and create a retriever for semantic search.

## 🧠 Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [HuggingFace Transformers](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [PDF Loader](https://docs.langchain.com/docs/integrations/document_loaders/pdf)
- Python

---

## ⚙️ Features

- ✅ Load and parse a PDF file
- ✅ Split text into overlapping chunks
- ✅ Embed using HuggingFace's `BAAI/bge-small-en-v1.5`
- ✅ Store embeddings in Chroma vector store
- ✅ Retrieve semantically similar chunks for question answering or further analysis

## 📁 Project Structure

```
project/ 
├── data/
│   └── chapter1.pdf         # Your PDF document
├── main.py                  # Main script
└── README.md                # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pdf-langchain-chroma.git
cd pdf-langchain-chroma
````

### 2. Install dependencies

```bash
pip install langchain langchain-community langchain-chroma langchain-huggingface chromadb
```

### 3. Run the script

```bash
python main.py
```

---

## 🧩 How It Works

### 1. Load PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('data/chapter1.pdf')
documents = loader.load()
```

### 2. Split into Chunks

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
```

### 3. Generate Embeddings and Store in Chroma

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
```

### 4. Setup Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)
```

---

## 🧠 Example Embedding Model Used

* **Model**: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
* **Use Case**: General-purpose embeddings for retrieval-based pipelines

---
