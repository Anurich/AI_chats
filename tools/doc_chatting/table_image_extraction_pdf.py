from transformers import AutoModelForObjectDetection
import torch
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt 
# import fitz
from pdf2image import convert_from_bytes
from torchvision import transforms
import cv2 
from utils.custom_logger import CustomLogger
import os 
import numpy as np
from utils import utility
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"
"""
    1. The idea is to create and langgraph 
    2. where the agent is going to call the Table extraction whenever the user ask the bot 
"""
# this is to resize the image 
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

class TableExtraction(CustomLogger):
    def __init__(self, pdfs:  List[str], path_for_image_and_text: str, language: str, client_s3, \
                 det_model, det_processor, rec_model, rec_processor):
        super().__init__(__name__)
        self.pdfs = pdfs
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
        self.id2label = self.model.config.id2label
        self.id2label[len(self.model.config.id2label)] = "no object"
        self.all_images: List = []
        self.prepared_images: List = []
        self.object_with_img_bbox: List[Dict] = [{}]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 4
        self.all_crop_images: List = []
        self.language = language
        self.client_s3 = client_s3
        self.path_for_image_and_text: str = path_for_image_and_text
        self.det_model, self.det_processor, self.rec_model, self.rec_processor = det_model, det_processor, rec_model, rec_processor
        
    def run_extraction(self, query: str):
        self.model.to(device) 
        # os.makedirs(self.path_for_image_and_text, exist_ok=True)
        # self.model_naught_ocr.to(device) # do not remove this it can be explored in future
        
        self.log_info("Starting pdf to image conversion.....")
        self.all_images, self.all_files = self._convert_pdf_page_to_image() # converting the pdf to images
        self.log_info("Image preperation for table detection.....")
        self.prepared_images = self._prepare_image_for_table_detection()
        self.log_info("Inference on images.....")
        self.object_with_img_bbox = self._perform_image_inference()
        self.all_crop_images, self.all_files_for_crop_image = self._crop_images()
        if len(self.all_crop_images) > 0:

            for idx, (cropped_img, filename_img) in enumerate(zip(self.all_crop_images, self.all_files_for_crop_image)):
                path_to_save_image =os.path.join(self.path_for_image_and_text,f"{filename_img.split(".pdf")[0]} Table {idx}.jpg")
                self.client_s3.write_data_as_image(cropped_img, path_to_save_image)

            # now we can perform the pddle ocr 
            all_tables = ""
            all_images = self.client_s3.s3_object_list(self.path_for_image_and_text)
            for idx, img in enumerate(tqdm(all_images)):
                if img.endswith("jpg"):
                    filename = img.replace(".jpg","")
                    path_to_write_txt = os.path.join(self.path_for_image_and_text, filename+".txt")
                    image_path = os.path.join(self.path_for_image_and_text, img)
                    df, df_markdown= utility.ocr_extraction(self.det_model, self.det_processor, \
                    self.rec_model, self.rec_processor,image_path, self.client_s3, lang = [self.language])
                    # now we can write it to txt file and also save the dataframe 
                    path_to_write_xlsx = os.path.join(self.path_for_image_and_text, filename+".xlsx")
                    self.client_s3.write_data_excel(df, path_to_write_xlsx)
                    all_tables += f"{filename}"+"\n"+df_markdown+"\n"
                    self.client_s3.write_data_as_txt(df_markdown,path_to_write_txt)
            
            # write the table
            path_to_write_all_files = os.path.join(self.path_for_image_and_text,"all_files_text.txt")
            self.client_s3.write_data_as_txt(all_tables, path_to_write_all_files)

        else:
            self.log_info("No images are available!")

    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # def _convert_pdf_page_to_image(self):
    #     zoom_x = 2.0  # horizontal zoom
    #     zoom_y = 2.0  # vertical zoom
    #     mat = fitz.Matrix(zoom_x, zoom_y)
    #     all_images = []
    #     self.log_info(f"Total PDFs to process: {len(self.pdfs)}")
    #     for pdf in tqdm(self.pdfs):
    #         pdf_bytes = self.client_s3.read_from_bucket(pdf["filename"])
    #         pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf") # load the pdf 
    #         for page_number in range(len(pdf_document)):
    #             page = pdf_document.load_page(page_number)
    #             pixmap = page.get_pixmap(matrix=mat)
    #             img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    #             all_images.append(img)
        
    #     self.log_info(f"Total images extracted from PDFs: {len(all_images)}")
    #     return all_images    

    def _convert_pdf_page_to_image(self):
        all_images = []
        all_files = []
        self.log_info(f"Total PDFs to process: {len(self.pdfs)}")
        
        for pdf in tqdm(self.pdfs):
            pdf_bytes = self.client_s3.read_from_bucket(pdf["filename"])
            # Convert PDF bytes to list of PIL images
            images = convert_from_bytes(pdf_bytes, fmt='jpg')
            # Append each page's image to all_images list
            all_images.extend(images)
            all_files.append(pdf["filename"])
        self.log_info(f"Total images extracted from PDFs: {len(all_images)}")
        return all_images, all_files
    

    def _prepare_image_for_table_detection(self):
        detection_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        prepared_images = []
        for img, filename in zip(self.all_images, self.all_files):
            org_img = img.copy()
            pixel_values = detection_transform(img).unsqueeze(0)
            pixel_values = pixel_values.to(device)
            prepared_images.append((pixel_values,org_img.size, org_img, filename))
        return prepared_images
    
    
    def _perform_image_inference(self):
        objects = []
        with torch.no_grad():
            for img, size, org_img, filename in tqdm(self.prepared_images):
                outputs = self.model(img) # inference form the image 
                m = outputs.logits.softmax(-1).max(-1)
                pred_labels = list(m.indices.detach().cpu().numpy())[0]
                pred_scores = list(m.values.detach().cpu().numpy())[0]
                pred_bboxes = outputs['pred_boxes'].detach().cpu()[0] 
                pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes,size)]
                for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
                    class_label = self.id2label[int(label)]
                    if class_label == 'no object':
                        continue 
                    elif float(score) > 0.95:
                        objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox],"image":org_img, "filename":filename})
        
        return objects

    def _crop_images(self):
        # Define the font properties
        all_crops_images = []
        all_filename = []
        if len(self.object_with_img_bbox) >=1:
            for object in self.object_with_img_bbox:
                image = np.array(object["image"])
                filename = object["filename"]
                bboxes = object["bbox"]
                bboxes = list(map(int, bboxes))
                x1, y1, x2, y2 = bboxes
                croped_image = image[y1: y2, x1:x2]
                all_crops_images.append(croped_image)
                all_filename.append(filename)
        else:
            self.log_warning("No tables detected in the images.")
        # let's combine the images
        return all_crops_images, all_filename
        # self._combine_images(all_crops_images)

    def _combine_images(self, image_list):
        num_images = len(image_list)
        num_rows = int(np.ceil(np.sqrt(num_images)))
        num_cols = int(np.ceil(num_images / num_rows))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        # If only one image, convert single Axes object to a sequence
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                index = i * len(ax_row) + j
                if index < num_images:
                    ax.imshow(image_list[index])
                    ax.axis('off')
                else:
                    ax.axis('off')  # Turn off axes if no image to display

        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
        plt.show()
