import os
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from utils import history, prompts
from langchain_community.callbacks import get_openai_callback
from utils.llm_cache import CacheInMemory
import time



class TableChat:
    def __init__(self,  file_path, client):
        self.table_file_path: str = file_path
        self.prompt: str = prompts.CHAT_WITH_TABLE
        self.template: PromptTemplate = PromptTemplate.from_template(self.prompt)
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=512)
        self.client = client
        self.llm_cache_in_memory = CacheInMemory()
        self.max_token_limit: int = 500
        self.chatHistory= history.chatHistory(max_token_limit=self.max_token_limit)
        
    def run_chat(self, query: str)-> List[Any]:
        with get_openai_callback()  as cb:
            start_time = time.time()
            if self.llm_cache_in_memory.check_cache(query) == None:
                path = os.path.join(self.table_file_path,"all_files_text.txt")
                texts = self.client.read_from_bucket(path).decode("utf-8")
                self.chatHistory.texts.append(texts)
                self.chatHistory.save_context_to_memory()

                chain = self.template | self.llm | StrOutputParser()
                filtered_list = [d for d in self.chatHistory.chat_history if 'summary' not in d]
                history = []
                history.append(texts)
                for idx, data in enumerate(filtered_list):
                    if idx % 2 !=0:
                        history.append(data["content"])

                output = chain.invoke({"history": history, "input": query})
                self.llm_cache_in_memory.add_to_cache(query, output)
                
            if self.llm_cache_in_memory.check_cache(query) != None:
                output = self.llm_cache_in_memory.check_cache(query)
            
            self.chatHistory.append_data_to_history(query, output)
            end_time = time.time()
            print(f"Total Time Taken: {end_time - start_time:0.2f}")
            print(f"{cb}")
        
        return self.chatHistory.chat_history, output
    