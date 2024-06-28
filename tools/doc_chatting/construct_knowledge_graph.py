from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from utils.custom_logger import CustomLogger

class KnowledgeGraph(CustomLogger):
    def __init__(self, llm):
        super().__init__(__name__)
        self.llm = llm 
        self.llm_graph = LLMGraphTransformer(llm = self.llm)
    
    def construct_knowledge_graph(self, response: str):
        self.log_info("Started the graph execution.")
        splitted_texts = response.split("Key Points:")[0]
        docs = [Document(page_content=splitted_texts)]
        graph_documents = self.llm_graph.convert_to_graph_documents(docs)
        return graph_documents