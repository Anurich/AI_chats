from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import utility, prompts
from collections import Counter
from langchain_openai import ChatOpenAI
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
        self.vector_db = Chroma.from_documents(self.recursive_texts, self.embedding_function, \
             persist_directory=os.path.join("/code/chat_with_pdf",persist_directory))

    def readVectorStore(self,persist_directory):
        """
            read the document from persis directory 
        """
        return Chroma(persist_directory = os.path.join("/code/chat_with_pdf", persist_directory), embedding_function = self.embedding_function)


class createVectorStore_DOC:
    def __init__(self, doc_object: dict, client, file_ids: dict,again=False):
        # now we can exrtact the pdfs
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.doc_object = doc_object
        self.file_ids  = file_ids
        self.client = client
        self.vector_storage = UTILS(self.doc_object)
        self.chain = PromptTemplate.from_template(prompts.CATEGORIZATION) | self.llm | StrOutputParser()
        self.chain_keyword = PromptTemplate.from_template(prompts.KEY_POINTS) | self.llm | StrOutputParser()
        # create a pdf to text
        self.categorization = dict()
        self.key_points = []
        self.create_pdf_texts()
        if len(self.page_texts) > 0:
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
        self.file_with_out_page_texts =[]
        for filename in self.doc_object.filenames:            
            temp_file_path = self.client.download_file_to_temp(filename)
            if filename.endswith("pdf"):
                file_uuid = self.file_ids[filename.split("/")[-1]]
                loader = PyPDFLoader(temp_file_path)
            if filename.endswith("txt"):
                loader = TextLoader(temp_file_path)
            
            document_chunked = loader.load_and_split()
            if len(document_chunked) != 0:
                for i in range(len(document_chunked)):
                    document_chunked[i].metadata = {
                        "source": filename,
                        "page": str(document_chunked[i].metadata["page"]),
                        "uuid": file_uuid
                    }
                
                categories = [{"Context": page.page_content} for page in tqdm(document_chunked)]
                outputs = self.chain.batch(categories)
                page_contents = [{"Context": data.page_content} for data in document_chunked]
                self.key_points = self.chain_keyword.batch(page_contents)[0]
                counts = Counter(outputs)
                category = counts.most_common(1)[0][0]
                if self.categorization.get(filename) == None:
                    self.categorization[filename] = category
                else:
                    self.categorization[filename].append(category)
                
                self.page_texts.extend(document_chunked)
            else:
                
                self.categorization[filename] = ["Others black"]
                print(self.categorization)

            os.remove(temp_file_path)
    
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