from langchain_community.document_loaders import UnstructuredFileLoader,TextLoader, Docx2txtLoader,UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import utility, prompts
from collections import Counter
from tqdm import tqdm
import re 
import os

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
        # page_texts_joined = [" ".join(page_texts)]
        self.recursive_texts = self.text_split.split_documents(page_texts)

    def createVectorStore(self, persist_directory) -> None:
        """
            Creating and Storing the vector store
        """
        for dir in persist_directory:
            self.vector_db = Chroma.from_texts(dir.page_content, self.embedding_function, persist_directory=persist_directory, collection_metadata=dir.metadata)

    def readVectorStore(self,persist_directory):
        """
            read the document from persis directory 
        """
        return Chroma(persist_directory = persist_directory, embedding_function = self.embedding_function)


class createVectorStore_DOC:
    def __init__(self, doc_object: dict, llm, client,again=False):
        # now we can exrtact the pdfs
        self.llm = llm
        self.doc_object = doc_object
        self.client = client
        self.vector_storage = UTILS(self.doc_object)
        self.chain = PromptTemplate.from_template(prompts.CATEGORIZATION) | self.llm | StrOutputParser()
        self.chain_keyword = PromptTemplate.from_template(prompts.KEY_POINTS) | self.llm | StrOutputParser()
        # create a pdf to text
        self.categorization = dict()
        self.key_points = []
        self.create_pdf_texts()
        # split the pdf into texts
        self.vector_storage.text_splitters(self.page_texts)
         # this is used for the computing the embedding 
        # if not os.path.isdir(self.doc_object.persist_directory):
        # if not os.path.isdir(self.doc_object.persist_directory):
        if not again:
            self.vector_storage.createVectorStore(self.doc_object.persist_directory)
        self.vector_db = self.vector_storage.readVectorStore(self.doc_object.persist_directory)

        
        
    def create_pdf_texts(self):
        """
            we flatten the pdf and create the one complete text 
            than we can split the pdf into multiple chunks with some overlap
        """
        
        self.page_texts = []
        
        for filename in self.doc_object.filenames:
            temp_file_path = self.client.download_file_to_temp(filename)
            if filename.endswith("pdf"):
                loader = UnstructuredFileLoader(temp_file_path)
            elif filename.endswith("txt"):
                loader = TextLoader(temp_file_path)
            elif filename.endswith("pptx"):
                loader = UnstructuredPowerPointLoader(temp_file_path)
            elif filename.endswith("docx"):
                loader = Docx2txtLoader(temp_file_path)
            
            document_chunked = loader.load_and_split()
            for i in range(len(document_chunked)):
                document_chunked[i].metadata = {
                    "filename": filename,
                    "page_number": i+1
                }
            
            outputs  = [self.chain.invoke({"Context": page.page_content}) for page in tqdm(document_chunked)]
            for page in tqdm(document_chunked):
                keypoint_output = self.chain_keyword.invoke({"Context":page.page_content})
                self.key_points.extend(keypoint_output.split("Key Points:")[1:])


            counts = Counter(outputs)
            category = counts.most_common(1)[0][0]
            if self.categorization.get(filename) == None:
                self.categorization[filename] = category
            else:
                self.categorization[filename].append(category)
            

            self.page_texts.extend(document_chunked)
        # now that we have the pdf_documents
        # we can combine the page_content form the pdf 
        # than we can create the text splitter     
        # for doc in pdf_docs:
            os.remove(temp_file_path)
        # self.page_texts = []
        # for doc in self.docs:
        #     self.page_texts.append(doc.page_content)

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