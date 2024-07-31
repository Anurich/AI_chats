import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.custom_logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from tqdm import tqdm
from langchain.load import dumps, loads
import re
import math
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
        assert len(file_paths) >= 1, self.log_error("Must have atleast 1 file !")
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
    
    def generate_html_table_with_graph(self, data):
        html = """
        <table border='1' style='border-collapse: collapse; width: 100%;'>
        <tr>
            <th>PDF Name</th>
            <th>Probability</th>
            <th>Page Number</th>
            <th>Context</th>
        </tr>
        """
        for i, (pdf_name, (probability, page_number, context)) in enumerate(data.items()):
            probability_percentage = probability * 100
            html += f"""
            <tr>
                <td>{pdf_name}</td>
                <td>
                    <canvas id='pieChart{i}' width='100' height='100'></canvas>
                    <div style='text-align: center; margin-top: 5px;'>{probability_percentage:.2f}%</div>
                </td>
                <td>{page_number}</td>
                <td>{context}</td>
            </tr>
            """
        
        html += """
        </table>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        document.addEventListener('DOMContentLoaded', function () {
            const chartsData = [
        """
        for i, (pdf_name, (probability, page_number, context)) in enumerate(data.items()):
            html += f"""
            {{
                label: 'Probability',
                data: [{probability * 100}, {100 - probability * 100}],
                backgroundColor: ['#4CAF50', '#ddd'],
                borderWidth: 1
            }},
            """
        
        html += """
            ];
            
            for (let i = 0; i < chartsData.length; i++) {
                const ctx = document.getElementById('pieChart' + i).getContext('2d');
                new Chart(ctx, {
                    type: 'pie',
                    data: {
                        datasets: [chartsData[i]]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            }
        });
        </script>
        """
        
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
        self.log_info("Sucessfully computred reranked-results.....")
        
        return reranked_results

    def search(self, description):
        response = self.vectordb_search.similarity_search(description, k= 100)
        relevance_score =dict()
        for content in tqdm(response):
            retriever = self.vectordb_search.as_retriever(search_kwargs={"k": 100})
            multi_query_generated = (ChatPromptTemplate.from_template(prompts.RAG_FUSION) | self.llm | StrOutputParser() | (lambda x: x.split("\n")))
            ragfusion_chain = multi_query_generated | retriever.map() | self.reciprocal_rank_fusion

            rag_output = ragfusion_chain.invoke({"question": description})
            all_outputs =[]
            for rg_doc, score in rag_output:
                print("---"*100)
                print(rg_doc)
                output = self.chain.invoke({"pdf_name": rg_doc.metadata["source"],"Context": rg_doc.page_content, "description": description})
                print(output)

                # pdf_name, probability, answer = output.split(":")
                # match = re.findall(r"[-+]?\d*\.\d+|\d+", probability)
                # assert len(match) == 1
                # if relevance_score.get(file_name) == None:
                #     relevance_score[file_name] = [float(match[0]), page_number, answer]
                # else:
                #     prob,_, _ = relevance_score[file_name]
                #     if prob < float(match[0]):
                #         relevance_score[file_name] = [float(match[0]), page_number, answer]        
            
        # html = self.generate_html_table_with_graph(relevance_score)
        # return html



    
