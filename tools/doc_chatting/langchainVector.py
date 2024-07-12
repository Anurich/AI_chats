from langchain_community.document_loaders import UnstructuredFileLoader,TextLoader, Docx2txtLoader,UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.s3_file import S3FileLoader
from utils import utility
import re 
from io import BytesIO
import os
import json

class UTILS:
    def __init__(self, doc_object: dict) -> None:
        self.text_split = RecursiveCharacterTextSplitter(chunk_size =doc_object.chunk_size, \
                                                chunk_overlap=doc_object.chunk_overlap, \
                                                length_function=len)
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
                                                
    def text_splitters(self, page_texts) -> None:
        """
            the idea is to split the texts into multiple chunks based on the chunk size and chunk overlap
        """
        page_texts_joined = [" ".join(page_texts)]
        self.recursive_texts = self.text_split.create_documents(page_texts_joined)

    def createVectorStore(self, persist_directory) -> None:
        """
            Creating and Storing the vector store
        """
        self.vector_db = Chroma.from_documents(self.recursive_texts, self.embedding_function, persist_directory=persist_directory)


    def readVectorStore(self,persist_directory):
        """
            read the document from persis directory 
        """
        return Chroma(persist_directory = persist_directory, embedding_function = self.embedding_function)


class createVectorStore_DOC:
    def __init__(self, doc_object: dict, llm client,again=False):
        # now we can exrtact the pdfs
        self.llm = llm
        self.doc_object = doc_object
        self.client = client
        self.vector_storage = UTILS(self.doc_object)
        # create a pdf to text
        self.create_pdf_texts()
        # split the pdf into texts
        self.vector_storage.text_splitters(self.page_texts)
         # this is used for the computing the embedding 
        # if not os.path.isdir(self.doc_object.persist_directory):
        # if not os.path.isdir(self.doc_object.persist_directory):
        if not again:
            self.vector_storage.createVectorStore(self.doc_object.persist_directory)
        self.vector_db = self.vector_storage.readVectorStore(self.doc_object.persist_directory)

        self.categorization = dict()
        
    def create_pdf_texts(self):
        """
            we flatten the pdf and create the one complete text 
            than we can split the pdf into multiple chunks with some overlap
        """
       
        self.docs = []
        assign_category = dict()
        for filename in self.doc_object.filenames:
            print(filename, "**"*100)
            temp_file_path = self.client.download_file_to_temp(filename)
            if filename.endswith("pdf"):
                loader = UnstructuredFileLoader(temp_file_path)
            elif filename.endswith("txt"):
                loader = TextLoader(temp_file_path)
            elif filename.endswith("pptx"):
                loader = UnstructuredPowerPointLoader(temp_file_path)
            elif filename.endswith("docx"):
                loader = Docx2txtLoader(temp_file_path)
            

            self.docs.extend(loader.load_and_split())
        # now that we have the pdf_documents
        # we can combine the page_content form the pdf 
        # than we can create the text splitter     
        # for doc in pdf_docs:
            os.remove(temp_file_path)
        self.page_texts = []
        for doc in self.docs:
            self.page_texts.append(doc.page_content)

class createVectorStore_WEB:
    def __init__(self, doc_object: dict) -> None:
        self.doc_object = doc_object
        web_data = utility.LoadFromWeb(self.doc_object.urls)
        self.vector_storage = UTILS(self.doc_object)
        
        self.page_texts = []
        for doc in web_data.docs:
            text = doc.page_content
            text = re.sub(r"\n+"," ",text)
            self.page_texts.append(text)
        
        self.vector_storage.text_splitters(self.page_texts)
        self.vector_storage.createVectorStore(self.doc_object.persist_directory)
        self.vector_db = self.vector_storage.readVectorStore(self.doc_object.persist_directory)