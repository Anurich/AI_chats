import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.custom_logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from tqdm import tqdm
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
            print(path)
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
        def generate_pie_chart_svg(probability):
            # Pie chart parameters
            radius = 10
            stroke_width = 2
            cx, cy = radius + stroke_width, radius + stroke_width
            r = radius - stroke_width / 2
            
            # Calculate the end angle
            end_angle = 360 * probability
            
            # SVG pie chart path data
            large_arc_flag = 1 if probability > 0.5 else 0
            end_x = cx + r * math.cos(math.radians(end_angle))
            end_y = cy - r * math.sin(math.radians(end_angle))
            path_d = (
                f"M {cx} {cy} "
                f"L {cx + r} {cy} "
                f"A {r} {r} 0 {large_arc_flag} 1 {end_x} {end_y} "
                f"Z"
            )
            
            return f"""
            <svg width="{2 * radius + stroke_width}" height="{2 * radius + stroke_width}" viewBox="0 0 {2 * radius + stroke_width} {2 * radius + stroke_width}">
                <circle cx="{cx}" cy="{cy}" r="{r}" fill="#ddd" />
                <path d="{path_d}" fill="#4CAF50" />
            </svg>
            """

        html = """
        <table border='1' style='border-collapse: collapse; width: 100%;'>
        <tr>
            <th>PDF Name</th>
            <th>Probability</th>
            <th>Page Number</th>
            <th>Context</th>
        </tr>
        """
        for pdf_name, (probability, page_number, context) in data.items():
            pie_chart_svg = generate_pie_chart_svg(probability)
            probability_percentage = probability * 100
            html += """
            <tr>
                <td>{}</td>
                <td>
                    {}
                    <div style='text-align: center; margin-top: 5px;'>{:.2f}%</div>
                </td>
                <td>{}</td>
                <td>{}</td>
            </tr>
        """.format(pdf_name, pie_chart_svg, probability_percentage, page_number, context)
        
        html += "</table>"
        return html

    def search(self, description):
        response = self.vectordb_search.similarity_search(description, k= 100)
        relevance_score =dict()
        for content in tqdm(response):
            metadata = content.metadata
            file_name = metadata["source"]
            page_number = metadata["page"]
            output = self.chain.invoke({"pdf_name": file_name,"Context": content.page_content, "description": description})
            pdf_name, probability, answer = output.split(":")
            match = re.findall(r"[-+]?\d*\.\d+|\d+", probability)
            assert len(match) == 1
            if relevance_score.get(file_name) == None:
                relevance_score[file_name] = [float(match[0]), page_number, answer]
            else:
                prob,_, _ = relevance_score[file_name]
                if prob < float(match[0]):
                    relevance_score[file_name] = [float(match[0]), page_number, answer]        
        
        print(relevance_score)
        html = self.generate_html_table_with_graph(relevance_score)
        return html



    
