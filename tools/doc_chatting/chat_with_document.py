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
from utils.custom_logger import CustomLogger

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
        #self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

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
        print("--"*100)
        print(reranked_results)
        print("--"*100)
        return reranked_results

    def run_chat(self, query: str):
        # summarize docs need not to be run everytime because if the document is summarized i can save it 
        # and next time when someone ask question i can simply use the saved one 
        retriever = self.vector_db.as_retriever(search_kwargs={"k": self.num_retrieved_docs})
        multi_query_generated = ( ChatPromptTemplate.from_template(prompts.RAG_FUSION) | self.llm | StrOutputParser() | (lambda x: x.split("\n")))
        ragfusion_chain = multi_query_generated | retriever.map() | self.reciprocal_rank_fusion

        rag_chain = (
            {"context": ragfusion_chain,  "question": itemgetter("question")} 
            | self.prompt 
            | self.llm
            | StrOutputParser() 
        )
        output = rag_chain.invoke({"question":query},config="metadata")
        print("*"*100,)
        print(output)
        self.chatHistory.append_data_to_history(query, output)
        #let's take always top last 5 in chat history 
        # to find the answer
        token_sentiment_response = self.sentiment_token_classification(self.llm, output)
        for token in token_sentiment_response[:-2]:
            output = output.replace(token.strip(), f"<<<<{token.strip()}>>>>")
        output += "\n **Sentiment:**\n "+token_sentiment_response[-1]

        return [output,  self.chatHistory.chat_history]

    def sentiment_token_classification(self, llm, content):
        """
        The function `sentiment_token_classification` prompts the user to perform token classification
        and sentiment analysis on provided content using a language model.
        
        :param llm: The `llm` parameter in the `sentiment_token_classification` function is likely
        referring to a language model (LLM) that is used for token classification and sentiment analysis
        tasks. This language model could be a pre-trained model like BERT, GPT-3, or any other model
        capable
        :param content: Based on the provided code snippet, it seems like you are working on a function
        that performs token classification and sentiment analysis on given content. The function takes in
        a language model (llm) and the content for analysis
        """
        self.log_info("Computing sentiment_token......")
        template = PromptTemplate.from_template(prompts.TOKEN_SENTIMENT_PROMPT)
        chain = template | llm | StrOutputParser()
        response = chain.invoke({"content": content})
        splitted_response = response.split("\n")[1:]
        response = list(map(lambda x: x.replace("-","").strip(),  splitted_response))
        response = list(filter(lambda x: x !="**Sentiment Analysis:**" and x!="Sentiment Analysis:" and x!='', response))
        self.log_info("Done..")
        return  response