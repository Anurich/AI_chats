import multiprocessing
from transformers import AutoModelForObjectDetection, AutoProcessor
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiProcessModelRunner:
    def __init__(self, model_name="microsoft/table-transformer-detection", revision="no_timm"):
        self.model_name = model_name
        self.revision = revision
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
        self.id2label = self.model.config.id2label
        self.id2label[len(self.model.config.id2label)] = "no object"
        
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _run_model(self, image_path, queue):
        # Load the model and processor (this is necessary in each process)
        # Load and preprocess the image
        
        inputs = self.processor(images=image, return_tensors="pt")

        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Put the results in the queue
        queue.put(outputs)

    def run_on_multiple_images(self, image_paths):
        # Create a queue to share results
        queue = multiprocessing.Queue()

        # Create a list to hold the process references
        processes = []

        # Create and start a process for each image
        for image_path in image_paths:
            p = multiprocessing.Process(target=self._run_model, args=(image_path, queue))
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results from the queue
        ren m

        return results