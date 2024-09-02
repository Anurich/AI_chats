from utils.prompts import SEMANTIC_CACHING
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
import os 

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
    def __init__(self, embedding_func,  user_id, type_of_chat):
        self.embedding_func = embedding_func
        self.counter =0
        self.chroma_client = chromadb.PersistentClient(path=f"filter_file_with_keyword/{user_id}_{type_of_chat}")
        self.collection = self.chroma_client.get_or_create_collection(name="db_with_file_search")


    def add_query_response(self, query, response):
        embedding = self.embedding_func.embed_query(query)
        self.collection.add(
            embeddings=[embedding],
            metadatas=[{"query": query, "response": str(response)}],
            ids=[str(self.counter)]
        )
        self.counter +=1

    def get_similar_response(self, query, threshold=0.20):
        new_embedding = self.embedding_func.embed_query(query)
        results = self.collection.query(
            query_embeddings=[new_embedding],
            n_results=1
        )
        
        nearest_distance = results["distances"][0]
        print(nearest_distance, "**"*10)
        if len(nearest_distance) > 0:
            if nearest_distance[0] < threshold:
                return results["metadatas"][0][0]["response"]
            else:
                return None
        else:
            return None



    def clear_cache(self):
        self.chroma_client.delete_collection(name="db_with_file_search")


    




