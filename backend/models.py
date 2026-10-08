from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


#   Models:
llm=ChatGroq(
    model='openai/gpt-oss-20b'
)

model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)