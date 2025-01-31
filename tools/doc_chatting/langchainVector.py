from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import utility, prompts
from langchain_core.documents import Document
import pytesseract
from collections import Counter
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
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

    def createVectorStore(self, persist_directory, metadata) -> None:
        """
            Creating and Storing the vector store
        """
     
        self.vector_db_chroma = Chroma.from_documents(self.recursive_texts, self.embedding_function, \
             persist_directory=os.path.join("/code/chat_with_pdf",persist_directory), collection_metadata=metadata)

    def readVectorStore(self,persist_directory):
        """
            read the document from persis directory 
        """
        return Chroma(persist_directory = os.path.join("/code/chat_with_pdf", persist_directory), embedding_function = self.embedding_function)


class createVectorStore_DOC:
    def __init__(self, doc_object: dict, client, chat_ids):
        # now we can exrtact the pdfs
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.doc_object = doc_object
        self.chat_ids  = chat_ids
        self.client = client
        self.vector_storage = UTILS(self.doc_object)
        self.chain = PromptTemplate.from_template(prompts.CATEGORIZATION) | self.llm | StrOutputParser()
        self.chain_keyword = PromptTemplate.from_template(prompts.KEY_POINTS) | self.llm | StrOutputParser()
        # create a pdf to text
        self.categorization = dict()
        self.key_points = []
        self.create_pdf_texts()
        if len(self.page_texts) > 0:
            self.metada_collections = None
            # split the pdf into texts
            self.vector_storage.text_splitters(self.page_texts)
            # this is used for the computing the embedding 
            # if not os.path.isdir(self.doc_object.persist_directory):
            # if not os.path.isdir(self.doc_object.persist_directory):
            self.vector_storage.createVectorStore(self.doc_object.persist_directory, self.metada_collections)
            # self.vector_db = self.vector_storage.readVectorStore(self.doc_object.persist_directory)
            self.vector_db = self.vector_storage.vector_db_chroma
            
    def delete_vectordb_from_chroma(self, metada_id, filename):
        ids_to_delete = []
        all_docs = self.vector_db._collection.get()
        for ids, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
            if metadata["uuid"] == metada_id and metadata["source"] == filename:
                ids_to_delete.append(ids)
        # # now we can delete it
        if len(ids_to_delete) > 0:
            self.vector_db._collection.delete(ids=ids_to_delete)

    def change_metadata(self,document_chunked, filename, table=False):
        if len(document_chunked) != 0:
            for i in range(len(document_chunked)):
                document_chunked[i].metadata = {
                    "source": filename,
                    "page": str(document_chunked[i].metadata["page"] + 1) if table ==False else "Table",
                    "uuid": self.chat_ids
                }
            # for vector db 
            self.metada_collections = {
                    "source": filename,
                    "uuid": self.chat_ids
                }
            return document_chunked
        return None

    def create_pdf_texts(self):
        """
            we flatten the pdf and create the one complete text 
            than we can split the pdf into multiple chunks with some overlap
        """
        
        self.page_texts = []
        document_chunkeds= dict() 
        pdf_file_name = None
        for filename in self.doc_object.filenames:
            temp_file_path = self.client.download_file_to_temp(filename)
            if filename.endswith("pdf"):
                pdf_file_name = filename
                
                loader = PyPDFLoader(temp_file_path)
                document_chunked = loader.load_and_split()
                chunked_docs = self.change_metadata(document_chunked, filename)
                if chunked_docs != None:
                    for i in range(len(chunked_docs)):
                        chunked_docs[i].page_content +=f" page number for this chunk is {chunked_docs[i].metadata["page"]}"
                    if document_chunkeds.get(pdf_file_name) == None:
                        document_chunkeds[pdf_file_name] = chunked_docs
                else:
                    # let's try pytesseract ocr so that we don't miss the record 
                    all_pages = convert_from_path(temp_file_path)
                    docs = []
                    if len(all_pages) > 0:
                        for idx, page in enumerate(all_pages):
                            text = pytesseract.image_to_string(page)
                            docs.append(Document(page_content=text +f" page number for this chunk is {idx+1}" , metadata={"source": filename, "page":idx+1,"uuid": self.chat_ids}))

                    document_chunkeds[pdf_file_name] = docs

            if filename.endswith("txt"):
                loader = TextLoader(temp_file_path)
                document_chunked = loader.load_and_split()
                chunked_docs = self.change_metadata(document_chunked, pdf_file_name,  table=True)
                if chunked_docs != None:
                    if document_chunkeds[pdf_file_name] == None:
                        document_chunkeds[pdf_file_name] = chunked_docs
                    else:
                        document_chunkeds[pdf_file_name].extend(chunked_docs)
            os.remove(temp_file_path)
            
        for filename, chunked_docs in document_chunkeds.items():
            if len(chunked_docs) > 0:
                categories = [{"Context": page.page_content} for page in tqdm(chunked_docs)]
                outputs = self.chain.batch(categories)
                page_contents = [{"Context": data.page_content} for data in chunked_docs]
                self.key_points = self.chain_keyword.batch(page_contents)[0]
                counts = Counter(outputs)
                category = counts.most_common(1)[0][0]
                if self.categorization.get(filename) == None:
                    self.categorization[filename] = category
                else:
                    self.categorization[filename].append(category)
                self.page_texts.extend(chunked_docs)
            else:
                self.categorization[filename] = "Others Black" 

    
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