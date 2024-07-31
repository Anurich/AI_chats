import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.custom_logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from utils import prompts

class Filesearchbykeyworddescrp(CustomLogger):
    def __init__(self, llm, client, persist_directory) -> None:
        super().__init__(__name__)
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        self.client =client
        self.llm = llm
        self.vectordb_search = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_function)
        self.text_split = RecursiveCharacterTextSplitter(chunk_size =2000, chunk_overlap=500, length_function=len)
        self.doc_id = 0
        self.prompt_file_search = PromptTemplate.from_template(prompts.FILE_SEARCH_PROMPT)
        self.chain = self.prompt_file_search | self.llm | StrOutputParser()

    def add_file_to_db(self, file_paths):
        self.log_info(f"Total of {len(file_paths)} files uploaded !")
        assert len(file_paths) == 1, self.log_error("Must have atleast 1 file !")
        for path in file_paths:
            file_path = path["filename"]
            temp_file_path = self.client.download_file_to_temp(file_path)
            self.log_info("File Download form bucket to temp folder!")
            if file_path.endswith("pdf"):
                loader = UnstructuredFileLoader(temp_file_path, mode="paged")
                chunked_document = loader.load_and_split()
                for i in range(len(chunked_document)):
                    chunked_document[i].metadata = {
                        "source": file_path.split("/")[1],
                        "page": str(chunked_document[i].metadata["page_number"])
                    }
                # we need to chunk it down to 
                recursive_texts = self.text_split.split_documents(chunked_document)
                all_chunks = [chunk.page_content for chunk in recursive_texts]
                all_ids = [str(i + self.doc_id) for i in range(len(all_chunks))]
                metadatas = [chunk.metadata for chunk in recursive_texts]                   
                self.vectordb_search.add_texts(
                    texts = all_chunks,
                    metadatas = metadatas,
                    ids=all_ids,
                )
                self.doc_id+= len(all_ids)
                self.log_info("Embedding stored successfully !")
            
            os.remove(temp_file_path)
            self.log_info("File removed from temp folder !")
    
    def search(self, description):
        response = self.vectordb_search.similarity_search(description, k= 100)
        relevance_score =dict()
        for content in tqdm(response):
            metadata = content.metadata
            file_name = metadata["source"]
            page_name = metadata["page"]
            output = self.chain.invoke({"Context": content.page_content, "description": description})
            print(output)
            pdf_name, probability = output.split(":")
            if relevance_score.get(pdf_name) == None:
                relevance_score[pdf] = [(probability, page_number)]
            else:
                relevance_score[pdf].append((probability, page_number))
        
        print(relevance_score)



    
