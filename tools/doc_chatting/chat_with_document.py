"""
    This is the chat modules
"""
import os 
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from utils import history, prompts
from typing import List
from langchain_community.vectorstores import Chroma
from typing import  List, Any
import json
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from utils.custom_logger import CustomLogger
from transformers import pipeline
import spacy

class Chatwithdocument(CustomLogger):
    def __init__(self, llm: ChatOpenAI, vector_db: Chroma):
        super().__init__(__name__)
        self.llm  = llm
        self.vector_db = vector_db
        self.num_docs_final: int = 30
        self.num_retrieved_docs: int = 20
        self.chat_history: List[Any] = []
        self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompts.CHAT_WITH_PDF)
        self.max_token_limit: int = 500
        self.chatHistory = history.chatHistory(max_token_limit=self.max_token_limit)
        #self.compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
        self.key = json.load(open("openai_keys/openai_cred.json", "r"))["API_COHERE_KEY"]
        self.nlp = spacy.load("en_core_web_lg")
    def reciprocal_rank_fusion(self, results: list[list], k=30):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}
        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
       
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        # Return the reranked results as a list of tuples, each containing the document and its fused score
        self.log_info("Sucessfully computred reranked-results.....")
        
        return reranked_results

    def run_chat(self, query: str):
        # Summarize docs need not to be run every time because if the document is summarized I can save it 
        # and next time when someone asks a question I can simply use the saved one 
        retriever = self.vector_db.as_retriever(search_kwargs={"k": self.num_retrieved_docs})
        multi_query_generated = (ChatPromptTemplate.from_template(prompts.RAG_FUSION) | self.llm | StrOutputParser() | (lambda x: x.split("\n")))
        ragfusion_chain = multi_query_generated | retriever.map() | self.reciprocal_rank_fusion

        rag_chain = (RunnablePassthrough.assign(context = (lambda x: x["context"]))
            | self.prompt 
            | self.llm
            | StrOutputParser() 
        )

        retriever_with_rag_fusion = (lambda x: x["question"]) | ragfusion_chain
        rag_chain_with_source = RunnablePassthrough.assign(context=retriever_with_rag_fusion).assign(answer = rag_chain)
        
        # Async processing
        response =  rag_chain_with_source.invoke({"question": query})

        output = response["answer"]
        self.chatHistory.append_data_to_history(query, output)
        
        # Let's take always top last 5 in chat history 
        # to find the answer
        
        output_answer = output.split("Sentiment:")[0].strip().replace("Answer:","")

        piepline_ner = pipeline(task="ner", model=self.model)
        ner_result = piepline_ner(output_answer)
        
        ner   = self.nlp(output_answer)
        tokens_with_label = []
        if ner.ents:
            for ner_obj in ner.ents:
                start_index = ner_obj.start_char
                end_index   = ner_obj.end_char
                label = ner_obj.label_
                text  = ner_obj.text
                tokens_with_label.append([start_index, end_index, label, text])
        
    
        sentiment = " ".join(output.split("Sentiment:")[1].split("Explanation:")).replace("\n","")
        output_answer += "\n **Sentiment:**\n "+sentiment
        
        max_count = 0
        metadata = None
        
        splitted_response = response["answer"].lower().split()
        for i in range(len(response["context"])):
            consider = 0
            doc,_ = response["context"][i]
            data = doc.page_content.lower()
            for res in splitted_response:
                if res in data:
                    consider +=1

            if consider > max_count:
                max_count  = consider
                metadata = doc.metadata
            
        return [output_answer+f" ***{metadata}*** ----{tokens_with_label}----",  self.chatHistory.chat_history]