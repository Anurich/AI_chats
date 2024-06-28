"""
LLM: we are going to use ollama mistral for this 
"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import os 
import json
from langchain import hub
from langchain_cohere import CohereRerank
from langchain_core.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

os.environ["AWS_MAX_ATTEMPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PATH_TO_OPENAI_KEY = "openai_keys/openai_cred.json"
OPENAI_API_KEY = json.load(open(PATH_TO_OPENAI_KEY, "r"))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY["API_key"]
os.environ["COHERE_API_KEY"] = OPENAI_API_KEY["API_COHERE_KEY"]
path_for_image_and_text="path_for_image_and_text"


class InitialiseRetreiver:
    def __init__(self, urls: List[str]) -> None:
        self.urls = urls
        self.num_retrieved_docs=10
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=256,chunk_overlap=0)
        documents = WebBaseLoader(urls).load()
        splitted_documents = self.splitter.split_documents(documents)
        self.compressor_cohere = CohereRerank()

        chromadb = Chroma.from_documents(
            documents = splitted_documents,
            collection_name="rag",
            embedding=OpenAIEmbeddings(),
            persist_directory="chromadb"
        )
        self.retriever = chromadb.as_retriever(search_kwargs={"k": self.num_retrieved_docs})
        

class grade:
    def __init__(self) -> None:
        llm =  ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

        prompt="""
           Given the query and the retrieved documents, please assess whether the documents are relevant and make sense in the context of the query. Here are the aspects to consider:
            1. **Relevance**: Do the documents contain information directly related to the query?

            2. **Accuracy**: Are the facts and information presented in the documents accurate and trustworthy?

            3. **Completeness**: Do the documents provide a comprehensive view of the topic mentioned in the query, or are they missing important details?

            4. **Clarity**: Is the information in the documents presented clearly and understandably?

            5. **Consistency**: Are the statements and information consistent throughout the documents?

            6. **Depth**: Does the depth of information in the documents match the complexity of the query?

            Please evaluate each document based on these criteria and provide your assessment:
            Query: {query}
            Retrieved Documents:{documents}
            For each document:
            - If the document is relevant, accurate, complete, clear, consistent, and deep in its information, please return "yes".
            - If the document is not relevant or lacks any of the mentioned criteria, please return "no".

            The output of yes and no should be in Json format with key name as score and output as [yes or no].
            I want only one single output as either yes or no with key as score.
        """
        prompt = PromptTemplate.from_template(prompt)
        self.chain = prompt | llm | JsonOutputParser()

    def invoke_grader(self, query: str, documents:List[str]) -> JsonOutputParser:
        response = self.chain.invoke({"query": query, "documents": documents})
        return response


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class Hallucination:
    def __init__(self) -> None:

        llm =  ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
        preamble = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
                Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. 
                The output of yes and no should be in Json format with key name as score and output as [yes or no].
                I want only one single output as either yes or no with key as score.
        """
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", preamble),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        self.chain = hallucination_prompt | llm | JsonOutputParser()

    def check_hallucination(self, generate_text: str, documents: List[str]) -> JsonOutputParser:
        print(generate_text)
        response = self.chain.invoke({"documents": documents, "generation": generate_text})
        return response

class QueryRewrite:
    def __init__(self):
        llm =  ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for web search. Look at the input and try to reason about the underlying sematic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
            ]
        )
        self.question_rewriter = re_write_prompt | llm | StrOutputParser()
    def generate_query(self, question):
        return self.question_rewriter.invoke({"question": question})

class GenerateQuery:
    def __init__(self):
        llm =  ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

        # Prompt
        preamble = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

        prompt = lambda x: ChatPromptTemplate.from_messages(
            [
                HumanMessage(
                    f"Question: {x['question']} \nAnswer: ",
                    additional_kwargs={"context": x["documents"]},
                )
            ]
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.rag_chain = prompt | llm | StrOutputParser()

    def generate(self, query, documents):
        generation = self.rag_chain.invoke({"documents": documents, "question": query})
        return generation

# hallucination = Hallucination()
# docs = ["hello my name is anupam", "i am 26", "i am sick"]
# output = hallucination.check_hallucination(generate_text="i am sick but i need to know about what is llm", documents=docs)
# print(output)

# query_formatter = QueryRewrite()
# print(query_formatter.generate_query("terms and condition"))


    