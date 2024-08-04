
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
        


