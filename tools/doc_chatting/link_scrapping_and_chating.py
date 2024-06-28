from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tools.doc_chatting.langchainVector import UTILS
from utils import history, prompts
from langchain_core.runnables import  RunnablePassthrough
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
import re 
from langchain_community.vectorstores import Chroma

class ChatWithWebsite:
    def __init__(self, llm: ChatOpenAI, vector_db: Chroma) -> None:
        self.llm  = llm
        self.vector_db = vector_db
        self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompts.CHAT_WITH_PDF)
        self.max_token_limit: int = 500
        self.num_retrieved_docs: int = 20
        self.chatHistory = history.chatHistory(max_token_limit=self.max_token_limit)
        self.compressor_cohere = CohereRerank()

    def run_web_chat_(self, query: str):
        retriever = self.vector_db.as_retriever(search_kwargs={"k": self.num_retrieved_docs})
        # docs =retriever.get_relevant_documents(query)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor_cohere, base_retriever=retriever
        )
        compressed_docs = compression_retriever.get_relevant_documents(query)
        data = [doc.page_content for doc in compressed_docs]
        self.chatHistory.append_data_to_history(data)
        rag_chain = (
            {"chat_history": lambda x: self.chatHistory.chat_history, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
        )
        output = rag_chain.invoke(query)
        self.chatHistory.append_data_to_history(output)
        #let's take always top last 5 in chat history 
        # to find the answer
        return [output, self.chatHistory.chat_history]