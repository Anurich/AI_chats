from langchain.tools import BaseTool, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Optional

class ChatWithTable(BaseTool):
    name="chat_with_table"
    description="This tool should be accessed when required to chat with table in pdf or without pdf."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None)-> str:
        return "chat_with_table"

class ChatWithPdf(BaseTool):
    name="chat_with_pdf"
    description="This tool should be used when we want to talk or chat with pdf. "
    def _run(self, query: str,run_manager: Optional[CallbackManagerForToolRun] = None)-> str:
        return "chat_with_pdf"

class TableExtraction(BaseTool):
    name="table_extraction"
    description="This tool is dedicated to extracting tables from PDF documents. It should be accessed only when there is a need for table extraction."
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None)-> str:
        return "table_extraction"

class WebsiteTalking(BaseTool):
    name = "chat_with_website"
    description = "useful when you have to talk to website through links, or when we have to scrap the data from website and started conversation with it."
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None)-> str:
        return "chat_with_website"

tool_names = ["chat_with_table", "chat_with_pdf", "table_extraction","chat_with_website"]
chatWithtable = ChatWithTable()
chatWithpdf = ChatWithPdf()
TableExtract = TableExtraction()
Webtalk = WebsiteTalking()
search = DuckDuckGoSearchRun()

tools_list = [
    Tool(name="chat_with_pdf", func=chatWithpdf.run, description="useful when we need to talk or chat with pdf."),
    Tool(name="chat_with_table", func=chatWithtable.run, description="useful when we need to talk or chat with table."),
    Tool(name="table_extraction", func=chatWithtable.run, description="This tool is dedicated to extracting tables from PDF documents. It should be accessed only when there is a need for table extraction."),
    Tool(name="Search",func=search.run,description="useful for when you need to answer questions about current events"),
    Tool(name="chat_with_website",func=Webtalk.run,description="useful when you have to talk to website through links, or when we have to scrap the data from website and started conversation with it.")
]