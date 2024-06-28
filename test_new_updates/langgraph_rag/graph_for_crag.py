from typing_extensions import TypedDict
from typing import List
from crag_test import InitialiseRetreiver,grade, Hallucination, QueryRewrite,GenerateQuery
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]
vectordb      = InitialiseRetreiver(urls)
grader        = grade()
hallucination = Hallucination()
write_query   = QueryRewrite()
generator     = GenerateQuery()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents 
    """
    question : str
    generation : str
    documents : List[str]
    chat_history: List[str]


state = GraphState()


def retreival_documents(state):
    question = state["question"]
    retriever = vectordb.retriever
    
    compression_retriever = ContextualCompressionRetriever(
            base_compressor=vectordb.compressor_cohere, base_retriever=retriever
        )

    compressed_docs = compression_retriever.get_relevant_documents(question)
    
    retreive_docs = [doc.page_content for doc in compressed_docs]
    return {"documents": retreive_docs}

def transform_query(state):
    question = state["question"]
    new_query = write_query.generate_query({"question":question})
    print("**"*10)
    print(new_query)
    return {"question": new_query}

def genrate_response(state):
    question = state["question"]
    documents  = state["documents"]
    output = generator.generate(question, documents)
    return {"generation": output}




def conditional_check_grader(state):
    documents = state["documents"]
    question  = state["question"]

    response_grader = grader.invoke_grader(question, documents)
    if response_grader["score"] == "no":
        return "transform_query"
    else:
        return "generate"
      
def condition_check_hallucination(state):
    generate = state["generation"]
    documents = state["documents"]
    print(generate)
    
    response = hallucination.check_hallucination(generate, documents)
    if response["score"] == "no":
        return "end"
    else:
        return "transform_query"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retreival_documents)  # retrieve
workflow.add_node("generate", genrate_response)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    conditional_check_grader,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)

workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)
# workflow.add_conditional_edges(
#     "generate",
#     condition_check_hallucination,
#     {
#         "transform_query": "transform_query",
#         "end":  END
#     }
# )

app = workflow.compile()

from pprint import pprint

# Run
inputs = {"question": "Types of CoT prompts"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])