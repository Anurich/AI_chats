from utils.prompts import SEMANTIC_CACHING
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb

class CacheInMemory:
    def __init__(self):
        self.cache_in_memory = dict()
    
    def check_cache(self, query):
        if query in self.cache_in_memory:
            return self.cache_in_memory[query]
        return None

    def add_to_cache(self, query, response):
        if self.cache_in_memory.get(query) == None:
            self.cache_in_memory[query]  = response
        


class SemanticMemory:
    def __init__(self, embedding_func, threshold=0.8):
        self.embedding_func = embedding_func
        self.threshold = threshold
        self.vectordb = chromadb.Client()
        

    def add_query_response():
        pass
    def check_cache(self, query):
        pass


    
        

    




