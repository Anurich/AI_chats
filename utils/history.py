from typing import List, Any, Dict
import json
import os 

class chatHistory:
    def __init__(self, max_token_limit: int):
        super().__init__()
        self. max_token_limit = max_token_limit
        self.texts =  []
        self.chat_history: List[Dict]=[]
        
    def save_context_to_memory(self):
        if any(self.texts):
            for cntx in self.texts:
                if len(self.chat_history) == 0:
                    self.chat_history.append({"summary": cntx})
                else:
                    contains_key_a = any("summary" in d for d in self.chat_history)
                    if not contains_key_a:
                        self.chat_history.append({"summary": cntx})

    def append_data_to_history(self,query, output):
        create_chat_history = [{
            "content": query,
        }, {
            "content": output
        }]

        self.chat_history.extend(create_chat_history)
        if len(self.chat_history) >10:
            del self.chat_history[0:2]
    
    def summarize_the_history(self):
        pass


