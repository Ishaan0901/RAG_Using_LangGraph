from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import model
from dotenv import load_dotenv
load_dotenv()


#   load the document:
book=PyPDFLoader('resources/harrypotter.pdf')
docs=book.load()

print(len(docs))



#   Chunking:
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)

print(len(chunks))



#   VectorStore:
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=model,
    persist_directory="VectorStore"
)

print("Vector store created successfully!")