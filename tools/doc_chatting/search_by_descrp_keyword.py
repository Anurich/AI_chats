import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.custom_logger import CustomLogger

class Filesearchbykeyworddescrp(CustomLogger):
    def __init__(self, client, persist_directory) -> None:
        super().__init__()
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        self.client =client
        self.vectordb_search = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_function)
        self.text_split = RecursiveCharacterTextSplitter(chunk_size =2000, chunk_overlap=500, length_function=len)
        self.doc_id = 0

    def add_file_to_db(self, file_paths):
        self.log_info(f"Total of {len(file_paths)} files uploaded !")
        assert len(file_paths) == 1, self.log_error("Must have atleast 1 file !")
        for file_path in file_paths:
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
                for chunk in recursive_texts:
                    doc_id = f"{file_path.split("/")[1].replace(".pdf","")}_page_{chunk.metadata['page']}_{self.doc_id}"
                    new_embedding = self.embedding_function.embed(chunk.page_content)

                    self.vectordb_search.add_documents(
                        documents=[{
                            "text": chunk.page_content,
                            "metadata": chunk.metadata
                        }],
                        ids=[doc_id],
                        embeddings=[new_embedding]
                    )
                    self.doc_id+=1

                self.log_info("Embedding stored successfully !")
            
            os.remove(temp_file_path)
            self.log_info("File removed from temp folder !")