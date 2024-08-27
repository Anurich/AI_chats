from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from utils.prompts import PDF_COMPARISION
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pytesseract
from pdf2image import convert_from_path


class PdfPreprocessingForComparision:
    def __init__(self, llm, client, doc_object) -> None:
        self.llm = llm 
        self.chain = PromptTemplate.from_template(PDF_COMPARISION) | self.llm | StrOutputParser()
        self.doc_object = doc_object
        self.client = client
        self.embeddings  = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, \
                                                chunk_overlap=0, \
                                                length_function=len)
        self.file1 = None
        self.file2 = None
        self.loader_file1_chunked  = None
        self.loader_file1_chunked = None
        for idx, filenames in enumerate(self.doc_object):
            if idx > 2:
                break
            temp_file_path = self.client.download_file_to_temp(filenames["filename"])
            if idx == 0:
                self.file1 = temp_file_path
            else:
                self.file2 = temp_file_path
        self.page_wise_text_file1 =None
        self.page_wise_text_file2 = None
        self.file_semantic_chunking()
        self.min_page = -1

    def read_through_pytesseract(self, temp_file_path):
        all_pages = convert_from_path(temp_file_path)
        docs = []
        if len(all_pages) > 0:
            for idx, page in enumerate(all_pages):
                text = pytesseract.image_to_string(page)
                docs.append(Document(page_content=text, metadata={"page":idx+1,"uuid": self.chat_ids}))

        return docs 

    def data_to_page_based_content(self, data):
        record = dict()
        for doc in data:
            if record.get(doc.metadata["page"]) == None:
                record[doc.metadata["page"]] = doc.page_content+" "
            else:
                record[doc.metadata["page"]] += doc.page_content
        
        return record
    
    def change_metadata(self,document_chunked):
        if len(document_chunked) != 0:
            for i in range(len(document_chunked)):
                document_chunked[i].metadata["page"] += 1
        return document_chunked        

    def file_semantic_chunking(self):
        self.loader_file1_chunked = PyPDFLoader(self.file1).load_and_split()
        self.loader_file2_chunked = PyPDFLoader(self.file2).load_and_split()
        if len(self.loader_file1_chunked) == 0:
            self.loader_file1_chunked = self.read_through_pytesseract(self.file1)
        

        if len(self.loader_file2_chunked) == 0:
            self.loader_file2_chunked = self.read_through_pytesseract(self.file2)
        

        # applying the semantic chunking 
        self.loader_file1_chunked = self.text_splitter.split_documents(self.loader_file1_chunked)
        self.loader_file2_chunked = self.text_splitter.split_documents(self.loader_file2_chunked)
    
        # page wise data 
        self.page_wise_text_file1 = self.data_to_page_based_content(self.loader_file1_chunked)
        self.page_wise_text_file2 = self.data_to_page_based_content(self.loader_file2_chunked)
        self.min_page = min(len(self.page_wise_text_file1.keys()), len(self.page_wise_text_file2))
        self.page_wise_comparision()

    def page_wise_comparision(self):
        self.response_with_page = dict()        
        for i in tqdm(range(int(self.min_page)+1)):
            if self.page_wise_text_file1.get(i) != None and self.page_wise_text_file2.get(i) != None:
                # perform the comparision between two same pages
                context1 = self.page_wise_text_file1[i]
                context2 = self.page_wise_text_file2[i]
                # summarize and than compare 
                response = self.chain.invoke({"pdf1": context1, "pdf2": context2})
                self.response_with_page[i] = response
            elif self.page_wise_text_file1.get(i) == None and self.page_wise_text_file2.get(i)!=None:
                for key, value in self.response_with_page.items():
                    if "extra" not in str(key):
                        context1 += value+" "
                context2 =self.page_wise_text_file2.get(i)
                response = self.chain.invoke({"pdf1": context1, "pdf2": context2})
                self.response_with_page[f"extra_{i}"] = response

            elif self.page_wise_text_file1.get(i) != None and self.page_wise_text_file2.get(i)==None:
                for key, value in self.response_with_page.items():
                    if "extra" not in str(key):
                        context2 += value+" "
                context1 =self.page_wise_text_file1.get(i)
                response = self.chain.invoke({"pdf1": context1, "pdf2": context2})
                self.response_with_page[f"extra_{i}"] = response
        



