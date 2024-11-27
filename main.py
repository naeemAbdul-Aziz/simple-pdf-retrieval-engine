from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#loading the documents
loader = PyPDFLoader('data/chapter1.pdf')
documents = loader.load()
print(documents)

#splitting and chunking the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(chunks)

#embedding, indexing and storing the documents
#intializing the OpenAIEmbeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)

# create a chroma vector store and embed the chunks
vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)

#instantiating a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# creating a prompt template
