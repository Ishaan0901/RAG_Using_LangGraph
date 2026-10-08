from typing import List, TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from langchain_chroma import Chroma
from models import llm,model
from dotenv import load_dotenv
import operator
load_dotenv()


#   State:
class BookState(TypedDict):
    question: str
    history: Annotated[List[str], operator.add]
    answer:str



def get_question(state: BookState):

    question = input("Human: ")

    return {
        "question": question
    }



#   RAG fucntion:
def RAG_pipeline(state: BookState):

    question=state['question']

    # Retriever
    store = Chroma(
        embedding_function=model,
        persist_directory="VectorStore"
    )

    retriever = store.as_retriever(
    search_kwargs={"k": 4}
    )
    context = retriever.invoke(question)

    # Prompt
    prompt = f'''
    You are a helpful assistant.

    Answer the user's question using only the
    information provided in the retrieved context.

    If the answer is not present in the context, say:
    "I don't know based on the provided book."

    Question:
    {question}

    History:
    {state["history"]}

    Context:
    {context}
    '''

    # LLM
    answer = llm.invoke(prompt).content
    print(f"AI: {answer}")

    return {
        "question": question,
        "history": [
            f"Human: {question}",
            f"AI: {answer}"
        ],
        "answer": answer
    }



def should_continue(state: BookState):

    if state["question"].lower() in ["exit", "bye", "quit"]:
        return END

    return "RAG"



graph = StateGraph(BookState)

graph.add_node("INPUT", get_question)
graph.add_node("RAG", RAG_pipeline)

graph.add_edge(START, "INPUT")

graph.add_conditional_edges(
    "INPUT",
    should_continue,
    {
        "RAG": "RAG",
        END: END
    }
)

graph.add_edge("RAG", "INPUT")

graph = graph.compile()

response=graph.invoke({
    "question": "",
    "history": [],
    "answer": ""
})
