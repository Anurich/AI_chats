import boto3
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import logging
from utils.custom_logger import CustomLogger

class BucketDigitalOcean(CustomLogger):
    """
    A class to interact with DigitalOcean Spaces for reading and writing data.
    
    Attributes:
        client_object (boto3.client): S3 client for DigitalOcean Spaces.
    """

    session = boto3.session.Session()
    client_object = session.client(
            's3',
            region_name='ap-south-1',
            aws_access_key_id='',
            aws_secret_access_key=''
        )
    def __init__(self):
        """
        Initializes the BucketDigitalOcean instance.
        """
        super().__init__(__name__)

    def _read_data(self, folder_path: str):
        """
        Internal method to read data from a specified path in the bucket.

        Args:
            folder_path (str): The path to the object in the bucket.

        Returns:
            dict: The response object from S3 containing metadata and body of the object.
                  Returns None if object does not exist or if there's an error.
        """
        try:
            response = self.client_object.get_object(Bucket="ai-server-bucket123", Key=folder_path)
            return response
        except self.client_object.exceptions.NoSuchKey as e:
            self.logger.error(f"Object '{folder_path}' does not exist. {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unable to read object '{folder_path}'. {e}")
            return None

    def read_from_bucket(self, folder_path: str):
        """
        Reads a file from the specified path in the bucket.
        Args:
            folder_path (str): The path to the file in the bucket.

        Returns:
            bytes: Bytes content of the  file.
                  Returns None if object does not exist or if there's an error.
        """
        response = self._read_data(folder_path)
        if response:
            return response['Body'].read()
        else:
            return None

    def read_image_from_bucket(self, folder_path: str) -> np.array:
        """
        Reads an image from the specified path in the bucket and returns it as a NumPy array.

        Args:
            folder_path (str): The path to the image file in the bucket.

        Returns:
            np.array: NumPy array representing the image.
                      Returns None if object does not exist or if there's an error.
        """
        response = self._read_data(folder_path)
        if response:
            image_bytes = response['Body'].read()
            pil_image = Image.open(BytesIO(image_bytes))
            numpy_array = np.array(pil_image)
            return numpy_array
        else:
            return None

    def write_data_as_image(self, img: np.array, folder_path: str):
        """
        Writes a NumPy array (representing an image) to the specified path in the bucket.

        Args:
            img (np.array): NumPy array representing the image.
            folder_path (str): The path to write the image file in the bucket.
        """
        try:
            pil_image = Image.fromarray(np.uint8(img))
            image_bytes = BytesIO()
            pil_image.save(image_bytes, format='JPEG')  # Adjust format as needed
            image_bytes.seek(0)

            self.client_object.put_object(Bucket='ai-server-bucket123',
                                          Key=folder_path,
                                          Body=image_bytes,
                                          ACL='private',
                                          Metadata={
                                              'x-amz-meta-my-key': 'your-value'
                                          }
                                          )
            self.logger.info(f"Image '{folder_path}' uploaded successfully.")
        except Exception as e:
            self.logger.error(f"Unable to write image '{folder_path}'. {e}")

    def write_data_excel(self, df: pd.DataFrame, folder_path: str):
        """
        Writes a Pandas DataFrame to the specified path in the bucket as an Excel file (XLSX).

        Args:
            df (pd.DataFrame): Pandas DataFrame to be written as Excel.
            folder_path (str): The path to write the Excel file in the bucket.
        """
        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            excel_buffer.seek(0)
            self.client_object.put_object(Bucket='ai-server-bucket123',
                                          Key=folder_path,
                                          Body=excel_buffer,
                                          ACL='private',
                                          Metadata={
                                              'x-amz-meta-my-key': 'your-value'
                                          }
                                          )
            self.logger.info(f"Excel file '{folder_path}' uploaded successfully.")
        except Exception as e:
            self.logger.error(f"Unable to write Excel file '{folder_path}'. {e}")

    def write_data_as_txt(self, data_object: str, folder_path: str):
        """
        Writes a text data object (string) to the specified path in the bucket.

        Args:
            data_object (str): Text data to be written.
            folder_path (str): The path to write the text file in the bucket.
        """
        try:
            self.client_object.put_object(Bucket='ai-server-bucket123',
                                          Key=folder_path,
                                          Body=data_object.encode("utf-8"),
                                          ACL='private',
                                          Metadata={
                                              'x-amz-meta-my-key': 'your-value'
                                          }
                                          )
            self.logger.info(f"Text file '{folder_path}' uploaded successfully.")
        except Exception as e:
            self.logger.error(f"Unable to write text file '{folder_path}'. {e}")

    def s3_object_list(self, folder_path: str):
        """
        Lists all JPEG files in a specified folder path in the bucket.

        Args:
            folder_path (str): The path to list objects from in the bucket.

        Returns:
            list: List of JPEG file names in the specified folder path.
                  Returns an empty list if no JPEG files are found or if there's an error.
        """
        try:
            response = self.client_object.list_objects_v2(Bucket="ai-server-bucket123")
            all_keys = [obj['Key'] for obj in response.get('Contents', [])]
            filtered_keys = [key.split('/')[-1] for key in all_keys if key.startswith(folder_path) and key.endswith('.jpg')]
            return filtered_keys
        except Exception as e:
            self.logger.error(f"Unable to list objects in folder '{folder_path}'. {e}")
            return []
    
    def s3_object_list_txt(self, folder_path: str):
        """
        Lists all TXT files in a specified folder path in the bucket.

        Args:
            folder_path (str): The path to list objects from in the bucket.

        Returns:
            list: List of TXT file names in the specified folder path.
                  Returns an empty list if no TXT files are found or if there's an error.
        """
        try:
            response = self.client_object.list_objects_v2(Bucket="ai-server-bucket123")
            all_keys = [obj['Key'] for obj in response.get('Contents', [])]
            filtered_keys = [key.split('/')[-1] for key in all_keys if key.startswith(folder_path) and key.endswith('.txt')]
            return filtered_keys
        except Exception as e:
            self.logger.error(f"Unable to list objects in folder '{folder_path}'. {e}")
            return []

    def download_file_to_temp(self, folder_path: str):
        """
        Downloads a file from the bucket to a temporary file path.

        Args:
            folder_path (str): The path to the file in the bucket to download.

        Returns:
            str: Temporary file path where the file was downloaded.
                 Returns None if download fails.
        """
        try:
            temp_file_path = tempfile.mktemp()
            self.client_object.download_file('ai-server-bucket123', folder_path, temp_file_path)
            self.logger.info(f"File '{folder_path}' downloaded to '{temp_file_path}'.")
            return temp_file_path
        except Exception as e:
            self.logger.error(f"Unable to download file '{folder_path}'. {e}")
            return None

    def remove_object_from_bucket(self, folder_path: str):
        """
        Removes an object (file) from the bucket.

        Args:
            folder_path (str): The path to the object (file) in the bucket to delete.
        """
        try:
            self.client_object.delete_object(Bucket="ai-server-bucket123", Key=folder_path)
            self.logger.info(f"Object '{folder_path}' deleted successfully.")
        except Exception as e:
            self.logger.error(f"Unable to delete object '{folder_path}'. {e}")

    def upload_vectordb_to_bucket(self, from_path, to_path):
        """
        Uploads a vector database file from a local path to an S3 bucket.

        :param from_path: str, local path to the file to be uploaded.
        :param to_path: str, destination path in the S3 bucket.
        :param bucket_name: str, name of the S3 bucket.
        """
        try:
            # Upload the file to S3
            self.client_object.upload_file(from_path, bucket_name, to_path)
            print(f"File uploaded successfully to s3://{bucket_name}/{to_path}")
        except Exception as e:
            print(f"Failed to upload file to S3: {e}")
