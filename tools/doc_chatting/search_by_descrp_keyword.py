import os
import json
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.custom_logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import  StrOutputParser, JsonOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from tqdm import tqdm
from langchain.load import dumps, loads
from concurrent.futures import ThreadPoolExecutor
from utils.llm_cache import SemanticMemory
import time
import re
import math
from utils import prompts



class Filesearchbykeyworddescrp(CustomLogger):
    def __init__(self, llm, client, persist_directory, user_id) -> None:
        super().__init__(__name__)
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm_cache_in_semantic_memory = SemanticMemory(embedding_func = self.embedding_function, user_id=user_id)
        self.client =client
        self.llm = llm
        self.vectordb_search = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_function)
        self.text_split = RecursiveCharacterTextSplitter(chunk_size =2000, chunk_overlap=500, length_function=len)
        self.doc_id = 0
        self.prompt_file_search = PromptTemplate(template = prompts.FILE_SEARCH_PROMPT, input_variables=["pdf_name", "Context", "description"])
        self.chain = self.prompt_file_search | self.llm | JsonOutputParser()

    def add_file_to_db(self, file_paths):
        assert len(file_paths) >= 1, f"At least have one file!"
        for file_path in tqdm(file_paths["filename"]):
            if file_path.endswith("pdf"):
                temp_file_path = self.client.download_file_to_temp(file_path)
                self.log_info("File Downloaded from bucket to temp folder!")
                loader = UnstructuredFileLoader(temp_file_path, mode="paged")
                chunked_document = loader.load_and_split()

                for i in tqdm(range(len(chunked_document))):
                    chunked_document[i].metadata = {
                        "source": file_path.split("/")[1],
                        "page": str(chunked_document[i].metadata["page_number"])
                    }

                # Chunk it down further
                recursive_texts = self.text_split.split_documents(chunked_document)
                all_chunks = [chunk.page_content for chunk in recursive_texts]
                all_ids = [str(i + self.doc_id) for i in range(len(all_chunks))]
                metadatas = [chunk.metadata for chunk in recursive_texts]
                
                # Adding texts to the vector database in batches
                self.vectordb_search.add_texts(
                    texts=all_chunks,
                    metadatas=metadatas,
                    ids=all_ids,
                )
                self.doc_id += len(all_ids)
                self.log_info("Embedding stored successfully!")

                os.remove(temp_file_path)
                self.log_info("File removed from temp folder!")


    def generate_html_table_with_graph(self,data):
        if not any(probability > 0.6 for probability, _, _, _ in data.values()):
            return "No file found with the given description."

        html = """
        <table border='1' style='border-collapse: collapse; width: 100%; class='file_filteration'>
        <tr>
            <th>PDF Name</th>
            <th>Probability</th>
            <th>Page Number</th>
            <th>Context</th>
            <th>Expected Answer</th>
        </tr>
        """
        for pdf_name, (probability, page_number, explaination, extracted_value) in data.items():
            if probability > 0.6:
                probability_percentage = probability * 100 
                html += f"""
                <tr>
                    <td>{pdf_name}</td>
                    <td>
                    <div style='background-color: #4CAF50; height: 20px; width: {probability_percentage}%;'></div>
                    <div style='text-align: center; margin-top: 5px;'>{probability_percentage:.2f}%</div>
                    </td>
                    <td>{page_number}</td>
                    <td>{explaination}</td>
                    <td>{extracted_value}</td>
                </tr>
                """
        html += "</table>"
        return html

    def reciprocal_rank_fusion(self, results: list[list], k=30):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}
        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
       
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        # Return the reranked results as a list of tuples, each containing the document and its fused score
        self.log_info("Sucessfully computed reranked-results.....")
        
        return reranked_results



    def search(self, description):
        with get_openai_callback()  as cb:
            
            start_time = time.time()
            cache_response = self.llm_cache_in_semantic_memory.get_similar_response(description)
            if cache_response == None:
                all_keys = ["pdf_name", "probability", "explanation", "expected_answer","page_number"]
                relevance_score =dict()
                retriever = self.vectordb_search.as_retriever(search_kwargs={"k": 20})
                multi_query_generated = (ChatPromptTemplate.from_template(prompts.RAG_FUSION) | self.llm | StrOutputParser() | (lambda x: x.split("\n")))
                ragfusion_chain = multi_query_generated | retriever.map() | self.reciprocal_rank_fusion

                rag_output = ragfusion_chain.invoke({"question": description})
                input_batch = []

                for rag_doc, score in tqdm(rag_output):
                    input_batch.append({"pdf_name": rag_doc.metadata["source"],"Context": rag_doc.page_content, "description": description,"page_number": rag_doc.metadata["page"]})
                
                outputs = self.chain.batch(input_batch)
                for output in tqdm(outputs):
                    # Rename 'explaination' to 'explanation' if it exists
                    if 'explaination' in output:
                        output['explanation'] = output.pop('explaination')

                    # Ensure all keys from 'all_keys' are present in the output
                    for key in all_keys:
                        if output.get(key) == None:
                            output[key] = "Not Found !"

                    # Extract relevant values
                    pdf_name = output["pdf_name"]
                    probability = float(output["probability"])
                    explanation = output["explanation"]
                    extracted_value = output["expected_answer"]

                    # Update relevance_score dictionary
                    if pdf_name not in relevance_score or relevance_score[pdf_name][0] < probability:
                        relevance_score[pdf_name] = [probability, output["page_number"], explanation, extracted_value]        

                html = self.generate_html_table_with_graph(relevance_score)
                    # we need to save into cache 
                self.llm_cache_in_semantic_memory.add_query_response(description, html)
            elif cache_response != None:
                html =  cache_response

            end_time = time.time()
            print(f"Total Time Taken: {end_time - start_time:0.2f}")
            print(f"{cb}")
            return html