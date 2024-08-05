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
        chroma_client = chromadb.Client()
        self.collection = chroma_client.get_or_create_collection(name="my_collection")


    def add_query_response(self, query, response):
        embedding = self.embedding_func.embed_query(query)
        self.collection.add(
            embeddings=[embedding],
            metadatas=[{"query": query, "response": response}],
            ids=[str(len(self.collection))]
        )
    def get_similar_response(self, query, threshold=0.5):
        new_embedding = self.embedding_func.embed_query(query)
        results = self.collection.(
            query_embeddings=[new_embedding],
            n_results=1
        )
        
        print(results)

    def clear_cache(self):
        self.collection.delete_all()


    




