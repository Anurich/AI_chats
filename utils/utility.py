from tqdm import tqdm
import numpy as np
import random
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import prompts
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any
import pandas as pd
import pickle
from PIL import Image
from surya.ocr import run_ocr
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from surya.model.detection import segformer
from langchain_openai import ChatOpenAI



def load_model_surya():
    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()
    return det_model, det_processor, rec_model, rec_processor

def creating_dataframe(all_texts: List[dict]):
    sorted_lines = sorted(all_texts, key=lambda x: x["bbox"][1])
    # Function to check if two bounding boxes are close enough to be on the same line
    def is_same_line(bbox1, bbox2, threshold=10):
        _, y_min1, _, y_max1 = bbox1
        _, y_min2, _, y_max2 = bbox2
        return abs(y_min1 - y_min2) <= threshold or abs(y_max1 - y_max2) <= threshold

    # Group lines based on their horizontal alignment
    formatted_lines = []
    current_line = []

    for i, line in enumerate(sorted_lines):
        if i == 0:
            current_line.append(line)
            continue
        
        if is_same_line(current_line[-1]["bbox"], line["bbox"]):
            current_line.append(line)
        else:
            formatted_lines.append(' '.join([text_line["text"] for text_line in current_line]))
            current_line = [line]

    # Add the last line
    if current_line:
        formatted_lines.append(' '.join([text_line["text"] for text_line in current_line]))

    # Create a DataFrame from the formatted lines
    df = pd.DataFrame(formatted_lines, columns=["Formatted Text"])
    return df, df.to_markdown()

def ocr_extraction(det_model, det_processor, rec_model, rec_processor,IMAGE_PATH, client_s3, lang: List[str]=["en"]):
    # image = Image.open(IMAGE_PATH)
    image = client_s3.read_image_from_bucket(IMAGE_PATH)
    image = Image.fromarray(image)
    # Replace with your languages
    predictions = run_ocr([image], [lang], det_model, det_processor, rec_model, rec_processor)
    all_texts = []
    for text in predictions[0].text_lines:
        new_texts = dict()
        new_texts["text"]=text.text
        new_texts["bbox"] = text.bbox
        all_texts.append(new_texts)
    df, df_markdown = creating_dataframe(all_texts)
    return df, df_markdown

def save_to_pickle(data: List[Any], filepath: str):
    with open(filepath, "wb") as fp:
        pickle.dump(data,fp)
    

def load_from_pickle(filepath: str) -> List[Any]:
    return pickle.load(open(filepath, "rb"))

def batched_summarize(documents, tokenizer, batch_size=16):
    # Process each document in the batch
    for i in range(0, len(documents), batch_size):
        batched_docs = documents[i: i+batch_size]
        all_texts = " ".join(batched_docs)
        tokenized_ids = tokenizer(all_texts, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        yield tokenized_ids

# for summary of the pdf 
def summarize_pdf(txt_file_path,keypoints, splitted_docs, client):
    """
        Approach we can use 
        First let's summarize the whole pdf using the opensource model 
        than we can use chatgpt api to find the bullet point int those summarization 
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response= None
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    #model.to_bettertransformer()
    documents_for_summarization = []
    non_summary = []
    splitted_docs = random.sample(splitted_docs, k=200) if len(splitted_docs) > 1000  else splitted_docs

    for doc in tqdm(splitted_docs):
        # Summarize the current document
        if len(doc.page_content.split()) > 100:
            # Append the summary to the list of all summaries
            documents_for_summarization.append(doc.page_content)   
        else:
            non_summary.append(doc.page_content)

    if any(documents_for_summarization):
        summary =[]
        for tokenized_ids in tqdm(batched_summarize(documents_for_summarization, tokenizer, batch_size=8)):
            
            output = model.generate(tokenized_ids["input_ids"])
            summary_output =tokenizer.decode(output[0], skip_special_tokens=True)
            summary.append(summary_output)
        all_summaries_text = ' '.join(summary)
    else:
        all_summaries_text = ' '.join(non_summary)

    # than we can create like a prompt to create an 
    chain= PromptTemplate.from_template(prompts.map_template) | llm | StrOutputParser()
    response = chain.invoke({"docs": all_summaries_text})
    keypoints.replace("Key Points:", "")
    response_with_keypoints = "Summary: \n"+ response +" \n KeyPoints: \n"+ keypoints
    client.write_data_as_txt(response_with_keypoints, txt_file_path)
    # client.write_data_as_txt()
    return response


def delete_vector_db(vectordb, file_id):
    print(vectordb.get())


def search_queries(content, queries):
    found_queries = []
    for query in queries:
        if query in content:
            found_queries.append(query)
            break
    return found_queries



class DotDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


class LoadFromWeb:
    def __init__(self, urls: List[str]) -> None:
        super().__init__()
        self.urls = urls
        self.loader = WebBaseLoader(self.urls)
        self.loader.requests_per_second = 1
        self.docs = self.loader.aload()