from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pytesseract
from pdf2image import convert_from_path


class PdfPreprocessingForComparision:
    def __init__(self, llm, client, doc_object) -> None:
        self.llm = llm 
        self.doc_object = doc_object
        self.client = client
        self.embeddings  = OpenAIEmbeddings()
        self.text_splitter = SemanticChunker(OpenAIEmbeddings(),add_start_index=True)
        self.file1 = None
        self.file2 = None
        self.loader_file1_chunked  = None
        self.loader_file1_chunked = None
        for idx, filename in enumerate(self.doc_object["file_names"]):
            if idx > 2:
                break
            temp_file_path = self.client.download_file_to_temp(filename)
            if idx == 0:
                self.file1 = temp_file_path
            else:
                self.file2 = temp_file_path


    def read_through_pytesseract(self, temp_file_path):
        all_pages = convert_from_path(temp_file_path)
        docs = []
        if len(all_pages) > 0:
            for idx, page in enumerate(all_pages):
                text = pytesseract.image_to_string(page)
                docs.append(Document(page_content=text, metadata={"page":idx+1,"uuid": self.chat_ids}))

        return docs 

    def file_semantic_chunking(self):
        self.loader_file1_chunked = PyPDFLoader(self.file1).load_and_split()
        self.loader_file2_chunked = PyPDFLoader(self.file2).load_and_split()

        if len(self.loader_file1_chunked) == 0:
            self.loader_file1_chunked = self.read_through_pytesseract(self.file1)
        elif len(self.loader_file2_chunked) == 0:
            self.loader_file1_chunked = self.read_through_pytesseract(self.file1)

        # applying the semantic chunking 
        self.loader_file1_chunked = self.text_splitter.split_documents(self.loader_file1_chunked)
        self.loader_file2_chunked = self.text_splitter.split_documents(self.loader_file2_chunked)
        