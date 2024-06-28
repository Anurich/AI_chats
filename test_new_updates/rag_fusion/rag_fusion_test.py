all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism.",
}
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import os
import json

os.environ["AWS_MAX_ATTEMPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PATH_TO_OPENAI_KEY = "openai_keys/openai_cred.json"
OPENAI_API_KEY = json.load(open(PATH_TO_OPENAI_KEY, "r"))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY["API_key"]
os.environ["COHERE_API_KEY"] = OPENAI_API_KEY["API_COHERE_KEY"]
path_for_image_and_text="path_for_image_and_text"


docs = [Document(page_content=doc, metadata={"source":"local"}) for doc in list(all_documents.values())]

chromadb = Chroma.from_documents(docs, collection_name="rag-fusion", embedding=OpenAIEmbeddings())
retriever = chromadb.as_retriever()


from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.load import dumps, loads


prompt = hub.pull("langchain-ai/rag-fusion-query-generation")

query_generator_chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

ragfusion_chain = query_generator_chain | retriever.map() | reciprocal_rank_fusion

print(ragfusion_chain.invoke({"original_query": "climate change"})) # Generate queries

