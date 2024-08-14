
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
import os
from typing import Dict
import uvicorn
from typing import List
import json
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from utils.bucket import BucketDigitalOcean
from spacy.cli import download
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from tools.doc_chatting.search_by_descrp_keyword import Filesearchbykeyworddescrp
download("en_core_web_lg")


####################
# move to seprate file 
app = FastAPI()
os.environ["AWS_MAX_ATTEMPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PATH_TO_OPENAI_KEY = "openai_keys/openai_cred.json"
OPENAI_API_KEY = json.load(open(PATH_TO_OPENAI_KEY, "r"))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY["API_key"]
os.environ["COHERE_API_KEY"] = OPENAI_API_KEY["API_COHERE_KEY"]
path_for_image_and_text="path_for_image_and_text"
llm =  ChatOpenAI(model="gpt-4o", temperature=0)
det_model, det_processor, rec_model, rec_processor = utility.load_model_surya()
all_user_vector_db  = dict()
all_user_table_chat =dict()
all_user_search_file=dict()
client = BucketDigitalOcean()
####################

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
    categories: List = []
    keyword_search: int =0
    



@app.post("/ai/model/chat_with_table")
async def chat_with_table(requestQuery: QueryRequest):
    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id
    if requestQuery.table_extract:
        tableExtraction = TableExtraction(pdfs=requestQuery.file_names,path_for_image_and_text=image_and_text_path,language=requestQuery.language,client_s3=client,\
                                          det_model=det_model, det_processor=det_processor,rec_model=rec_model, rec_processor=rec_processor)
        tableExtraction.run_extraction("") # this tool need to be run in order to save the images into the files 
        response = {
            "output": "Extraction Complete!",
            "table_extract": False,
        }
    else:
        if all_user_table_chat.get(ids) != None:
            chat_history, output  = all_user_table_chat[ids].run_chat(requestQuery.query)
        elif all_user_table_chat.get(ids) == None:
            chatwithtable = TableChat( file_path=image_and_text_path, client=client)
            all_user_table_chat[ids] = chatwithtable
            chat_history, output = all_user_table_chat[ids].run_chat(requestQuery.query)

        
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
    image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id+"/"
    responses = client.s3_object_list_txt(image_and_text_path)
    all_file_names=[]
    file_ids = dict()
    for files in requestQuery.file_names:
        txt_file = list(filter(lambda x: files["filename"] in x , responses))
        if len(txt_file) > 0:
            txt_file_path= requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id+"/"+txt_file[0]
            all_file_names.extend([files["filename"], txt_file_path])
        else:
            all_file_names.append(files["filename"])
        file_ids[files["filename"].split("/")[-1]] = files["base_64_content"]

    print(all_file_names, "**"*10)
    config.file_config["chat_with_pdf"]["filenames"] = all_file_names
    config.file_config["chat_with_pdf"]["persist_directory"] = "chromadb/"+requestQuery.user_id+"_"+requestQuery.chat_id+"_chromadb"
    
    object_chat_with_pdf = utility.DotDict(config.file_config["chat_with_pdf"])
    print(all_file_names, "**"*100)
    vector_doc = createVectorStore_DOC(object_chat_with_pdf,client,file_ids)
    chat_tool = Chatwithdocument(vector_db=vector_doc.vector_db if len(vector_doc.page_texts) > 0 else "" ,llm=llm, user_id=requestQuery.user_id)
  
    SAVE_SUMMAIZE_DIR = f"{requestQuery.path_for_summarization}/{requestQuery.user_id}_{requestQuery.chat_id}/"
    if os.path.isdir(SAVE_SUMMAIZE_DIR):
        shutil.rmtree(SAVE_SUMMAIZE_DIR)
    
    save_file_path = os.path.join(SAVE_SUMMAIZE_DIR,"summary.txt")
    if len(vector_doc.page_texts) >0:
        output_summary = utility.summarize_pdf(save_file_path,vector_doc.key_points,vector_doc.vector_storage.recursive_texts, client)
    else:
        output_summary ="1. All Pages are image in the pdf! Please check the table sections! \n"



    all_user_vector_db[ids] = [vector_doc, output_summary, chat_tool]
    response= {
        "summary": output_summary,
        "chat_id": requestQuery.chat_id,
        "intermediate_steps": ["chat_with_pdf"],
        "categories": [vector_doc.categorization]
    }

    print(response)

    return response



@app.post("/ai/model/chat_with_pdf")
async def chat_with_pdf(requestQuery: QueryRequest):
    # once i am here i need to respond with firs the summary of the pdf
    # for file 
    vector_db = None
    chat_tool = None
    summary = None
    

    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    
    # if all_user_vector_db.get(ids) == None:
    #     # image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id+"/all_files_text.txt"
    #     summary_path = f"{requestQuery.path_for_summarization}/"+requestQuery.user_id+"_"+requestQuery.chat_id+"/summary.txt"
    #     data = client.read_from_bucket(summary_path).decode("utf-8")
    #     # print( "chromadb/"+requestQuery.user_id+"_"+requestQuery.chat_id+"_chromadb")
    #     config.file_config["chat_with_pdf"]["filenames"] = ""
    #     config.file_config["chat_with_pdf"]["persist_directory"] = "chromadb/"+requestQuery.user_id+"_"+requestQuery.chat_id+"_chromadb"
    #     object_chat_with_pdf = utility.DotDict(config.file_config["chat_with_pdf"])
    #     file_ids = dict()
    #     for files in requestQuery.file_names:
    #         all_file_names.append(files["filename"])
    #         file_ids[files["filename"].split("/")[-1]] = files["base_64_content"]
    #     vector_doc = createVectorStore_DOC(object_chat_with_pdf,client,again=True)
    #     chat_tool = Chatwithdocument(vector_db=vector_doc.vector_db,llm=llm, user_id=requestQuery.user_id)
    #     all_user_vector_db[ids] = [vector_db, data, chat_tool]

    if all_user_vector_db.get(ids) != None:
        vector_doc,  summary, chat_tool = all_user_vector_db[ids]
        output,  chat_history =  chat_tool.run_chat(requestQuery.query)
    
    else:
        output = "Please summarize the file before!"
        chat_history = [output]
    

    
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

@app.post("/ai/model/search_keyword")
async def file_search(requestQuery: QueryRequest):
    file_search_by_keyword = Filesearchbykeyworddescrp(llm=llm, client=client, persist_directory="search_by_keyword",user_id=requestQuery.user_id)
    if requestQuery.keyword_search == 0:
        if all_user_search_file.get(requestQuery.user_id) == None:
            file_search_by_keyword.add_file_to_db(requestQuery.file_names)
            all_user_search_file[requestQuery.user_id] = file_search_by_keyword
        else:
            all_user_search_file[requestQuery.user_id].add_file_to_db(requestQuery.file_names)
    elif requestQuery.keyword_search == 1:
        print(all_user_search_file)
        response = all_user_search_file[requestQuery.user_id].search(requestQuery.query)
        return {
            "output": response,
            "chat_id": requestQuery.chat_id,
            "query": requestQuery.query,
        }

            


@app.post("/ai/model/router")
async def router(requestQuery: QueryRequest):
    image_and_text_path = requestQuery.path_for_image_and_text+"/"+requestQuery.user_id+"/"+requestQuery.chat_id
    template = PromptTemplate.from_template(prompts.ROUTER)
    chain = template | llm | JsonOutputParser()
    all_images = client.s3_object_list(image_and_text_path)
    json_output = chain.invoke({"query": requestQuery.query, "table": str(len(all_images))})
    return json_output



@app.post("/ai/model/delete_vectordb")
async def delete_vector_db(requestQuery: QueryRequest):
    ids = f"{requestQuery.user_id}_{requestQuery.chat_id}"
    db, _, _ = all_user_vector_db[ids]


    file_ids = []
    for files in requestQuery.file_names:
        file_ids.append(files["base_64_content"])
    assert len(file_ids) == 1
    utility.delete_vector_db(db,file_ids[0])



# all_ids_mapping = dict()
# @app.post("/ai/model/run")
# async def normal_agent_chat(requestQuery: QueryRequest):
#     chat_history = None
#     memory = None
#     ids = requestQuery.user_id+"_"+requestQuery.chat_id
#     if all_ids_mapping.get(ids) != None:
#         chat_history = all_ids_mapping[ids][0]
#         memory = all_ids_mapping[ids][1]
#     elif all_ids_mapping.get(ids) == None:
#         chat_history = MessagesPlaceholder(variable_name="chat_history")
#         memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
#         memory.output_key = "output"
#         all_ids_mapping[ids] = [chat_history, memory]

#     agent = create_openai_functions_agent(llm, tools_structure.tools_list, prompts.MAIN_PROMPT)
#     agent_executor = AgentExecutor(agent=agent, 
#                     tools=tools_structure.tools_list, 
#                     verbose=True,
#                     handle_parsing_errors=True,
#                     memory = memory,
#                     return_intermediate_steps = True,
#                     )
#     output = agent_executor.invoke({"input":requestQuery.query, "chat_history":chat_history}) 


#     if len(output["intermediate_steps"]) != 0:
#         agent_action, _ = output["intermediate_steps"][0]
#         tool_name = agent_action.tool
#         output["intermediate_steps"] = [tool_name]

#     output["chat_id"] = requestQuery.chat_id
#     return output

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4200)
