
from configuration import config
from tools.doc_chatting.langchainVector import createVectorStore_DOC, createVectorStore_WEB
from tools.doc_chatting.chat_with_document import Chatwithdocument
from tools.doc_chatting.table_image_extraction_pdf import TableExtraction
from tools.doc_chatting.construct_knowledge_graph import KnowledgeGraph
from tools.doc_chatting.talk_to_table import TableChat
from tools.doc_chatting.link_scrapping_and_chating import ChatWithWebsite
from langchain.prompts import  PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils import utility, prompts
import shutil
from utils.doc_chatting import tools_structure
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
import os
from typing import Dict
import uvicorn
from langchain.prompts import MessagesPlaceholder
from typing import List
import json
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import multiprocessing
from utils.bucket import BucketDigitalOcean
import asyncio

app = FastAPI()
os.environ["AWS_MAX_ATTEMPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PATH_TO_OPENAI_KEY = "openai_keys/openai_cred.json"
OPENAI_API_KEY = json.load(open(PATH_TO_OPENAI_KEY, "r"))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY["API_key"]
os.environ["COHERE_API_KEY"] = OPENAI_API_KEY["API_COHERE_KEY"]
path_for_image_and_text="path_for_image_and_text"



llm =  ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)

class QueryRequest(BaseModel):
    query: str 
    chat_history: List[str]
    output: str
    intermediate_steps: List[str]
    file_names: List[Dict[str, str]]
    user_id: str
    chat_id: str
    path_for_image_and_text: str = "file_uploads"
    path_for_summarization: str = "file_uploads/summarize_doc_into_txt"
    table_extract:bool=False
    website_link:str= None
    language: str = "en"
    node:List =[]
    relationship:List = []
    

all_user_vector_db = dict()
all_user_table_chat =dict()
client = BucketDigitalOcean()

@app.post("/ai/model/chat_with_table")
async def chat_with_table(requestQuery: QueryRequest):
    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id
    if requestQuery.table_extract:
        tableExtraction = TableExtraction(pdfs=requestQuery.file_names,path_for_image_and_text=image_and_text_path,language=requestQuery.language,client_s3=client)
        tableExtraction.run_extraction("") # this tool need to be run in order to save the images into the files 
        response = {
            "output": "Extraction Complete!",
            "table_extract": False,
        }
    else:
        if all_user_table_chat.get(ids) != None:
            chat_history, output  = all_user_table_chat[ids].run_chat(requestQuery.query)
        elif all_user_table_chat.get(ids) == None:
            chatwithtable = TableChat(llm=llm, file_path=image_and_text_path, client=client)
            all_user_table_chat[ids] = chatwithtable
            chat_history, output = all_user_table_chat[ids].run_chat(requestQuery.query)

        
        response_graph =[]
        response =  {
            "query": requestQuery.query,
            "chat_history": chat_history,
            "output": output,
            "chat_id": requestQuery.chat_id,
            "node": [],
            "relationship": [],
            "table_extract": False,
            "intermediate_steps":["chat_with_table"]
        }
    
    return response

@app.post("/ai/model/summarize")
async def summarization_doc(requestQuery: QueryRequest):

    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id+"/all_files_text.txt"
    all_file_names=[]

    for files in requestQuery.file_names:
        all_file_names.append(files["filename"])

    responses = client.s3_object_list(image_and_text_path)
    if len(responses) >0:
        txt_file =requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id+"/all_files_text.txt"
        all_file_names.append(txt_file)

    config.file_config["chat_with_pdf"]["filenames"] = all_file_names
    config.file_config["chat_with_pdf"]["persist_directory"] = "chromadb/"+requestQuery.user_id+"_"+requestQuery.chat_id+"_chromadb"
    
    object_chat_with_pdf = utility.DotDict(config.file_config["chat_with_pdf"])
    vector_doc = createVectorStore_DOC(object_chat_with_pdf,client)
    chat_tool = Chatwithdocument(vector_db=vector_doc.vector_db,llm=llm)

  
    SAVE_SUMMAIZE_DIR = f"{requestQuery.path_for_summarization}/{requestQuery.user_id}_{requestQuery.chat_id}/"
    if os.path.isdir(SAVE_SUMMAIZE_DIR):
        shutil.rmtree(SAVE_SUMMAIZE_DIR)
    
    save_file_path = os.path.join(SAVE_SUMMAIZE_DIR,"summary.txt")

    output_summary = utility.summarize_pdf(llm,save_file_path,vector_doc.vector_storage.recursive_texts, client)
    all_user_vector_db[ids] = [vector_doc, output_summary, chat_tool]
    response= {
        "summary": output_summary,
        "chat_id": requestQuery.chat_id,
        "intermediate_steps": ["chat_with_pdf"]
    }
    return response



@app.post("/ai/model/chat_with_pdf")
async def chat_with_pdf(requestQuery: QueryRequest):
    # once i am here i need to respond with firs the summary of the pdf
    # for file 
    vector_db = None
    chat_tool = None
    summary = None
    

    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    
    if all_user_vector_db.get(ids) == None:
        # image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id+"/all_files_text.txt"
        summary_path = f"{requestQuery.path_for_summarization}/"+requestQuery.user_id+"_"+requestQuery.chat_id+"/summary.txt"
        data = client.read_from_bucket(summary_path).decode("utf-8")
        # print( "chromadb/"+requestQuery.user_id+"_"+requestQuery.chat_id+"_chromadb")
        config.file_config["chat_with_pdf"]["filenames"] = ""
        config.file_config["chat_with_pdf"]["persist_directory"] = "chromadb/"+requestQuery.user_id+"_"+requestQuery.chat_id+"_chromadb"
        object_chat_with_pdf = utility.DotDict(config.file_config["chat_with_pdf"])
        vector_doc = createVectorStore_DOC(object_chat_with_pdf,client,again=True)
        chat_tool = Chatwithdocument(vector_db=vector_doc.vector_db,llm=llm)
        all_user_vector_db[ids] = [vector_db, data, chat_tool]

    if all_user_vector_db.get(ids) != None:
        vector_doc,  summary, chat_tool = all_user_vector_db[ids]

    output,  chat_history = chat_tool.run_chat(requestQuery.query)
    # response_graph = graph.construct_knowledge_graph(output)
    response_graph=[]
   
    
    return {
        "query": requestQuery.query,
        "chat_history": chat_history,
        "output": output,
        "summarize": summary,
        "chat_id": requestQuery.chat_id,
        "node": [],
        "relationship": [],
        "intermediate_steps":["chat_with_pdf"]
    }

@app.post("/ai/model/knowledge_graph")
async def knowledge_graph_computation(requestQuery : QueryRequest):
    graph  = KnowledgeGraph(llm)
    histories = requestQuery.chat_history
    graph_texts = []
    for history in histories:
        history = history.replace("<<<<", "").replace(">>>>", "")
        graph_texts.append(history)
    if len(graph_texts) == 1:
        kg = graph.construct_knowledge_graph(graph_texts[0])
    else:
        texts = " ".join(graph_texts)
        kg = graph.construct_knowledge_graph(texts)
    

    return {
        "node": kg[0].nodes,
        "relationship": kg[0].relationships,
        "chat_id": requestQuery.chat_id
    }

@app.post("/ai/model/chat_with_website")
async def chat_with_website(requestQuery: QueryRequest):

    if requestQuery.website_link != None:
        SAVE_SUMMAIZE_DIR = "summarize_web_data_in_txt"
        os.makedirs(SAVE_SUMMAIZE_DIR, exist_ok=True)
        save_file_path = os.path.join(SAVE_SUMMAIZE_DIR,"summary.txt")
        output_summary = utility.summarize_pdf(llm,save_file_path,chat_website.vs.recursive_texts)
        
    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    if all_user_vector_db.get(ids) == None:
        config.file_config["web-loader"]["urls"] = [requestQuery.website_link]
        object_web_scrapping_link =utility.DotDict(config.file_config["web-loader"])
        vector_web = createVectorStore_WEB(object_web_scrapping_link)
        chat_website = ChatWithWebsite(llm,vector_web.vector_db)
        all_user_vector_db[ids] = [vector_web, chat_website]
    elif all_user_vector_db.get(ids) != None:
        vector_web, chat_website = all_user_vector_db.get(ids)

    output, chat_history = chat_website.run_web_chat_(requestQuery.query)
    return {
        "query": requestQuery.query,
        "chat_history": chat_history,
        "output": output
    }

@app.post("/ai/model/router")
async def router(requestQuery: QueryRequest):
    template = PromptTemplate.from_template(prompts.ROUTER)
    chain = template | llm | JsonOutputParser()
    json_output = chain.invoke({"query": requestQuery.query})
    return json_output



all_ids_mapping = dict()
@app.post("/ai/model/run")
async def normal_agent_chat(requestQuery: QueryRequest):
    chat_history = None
    memory = None
    ids = requestQuery.user_id+"_"+requestQuery.chat_id
    if all_ids_mapping.get(ids) != None:
        chat_history = all_ids_mapping[ids][0]
        memory = all_ids_mapping[ids][1]
    elif all_ids_mapping.get(ids) == None:
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        memory.output_key = "output"
        all_ids_mapping[ids] = [chat_history, memory]

    agent = create_openai_functions_agent(llm, tools_structure.tools_list, prompts.MAIN_PROMPT)
    agent_executor = AgentExecutor(agent=agent, 
                    tools=tools_structure.tools_list, 
                    verbose=True,
                    handle_parsing_errors=True,
                    memory = memory,
                    return_intermediate_steps = True,
                    )
    output = agent_executor.invoke({"input":requestQuery.query, "chat_history":chat_history}) 


    if len(output["intermediate_steps"]) != 0:
        agent_action, _ = output["intermediate_steps"][0]
        tool_name = agent_action.tool
        output["intermediate_steps"] = [tool_name]

    output["chat_id"] = requestQuery.chat_id
    return output


from pathlib import Path
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4200)