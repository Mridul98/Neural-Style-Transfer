import logging
import json
from PIL import Image
from minio import Minio
from minio.error import S3Error
from io import BytesIO
logging.basicConfig(level=logging.INFO,force=True)

class ImageStorage:

    def __init__(self,host:str,access_key:str,secret_key:str,bucket_name:str):
        
        self.client_instance = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        self.bucket_name = bucket_name
        logging.info(f'minio client instance id: {id(self.client_instance)}')

    def check_connection(self):
        try:
            self.client_instance.bucket_exists(self.bucket_name)
        except Exception as e:
            logging.error(f'error connecting to minio {e}')
            return False
        return True
    
    def list_files(self,prefix:str=None):
        try:
            objects = self.client_instance.list_objects(self.bucket_name,prefix=prefix)
        except S3Error as e:
            logging.error(f'error listing files from minio. Error message: {e}')
            return None
        return objects
    
    def get_file_stats(self,object_name:str):
        try:
            stats = self.client_instance.stat_object(self.bucket_name,object_name)
        except S3Error as e:    
            logging.error(f'error getting file stats {object_name}. Error message: {e}')
            return None
        return stats.metadata
    
    def get_file(self,file_type:str, object_name:str):

        response = None
        file = None
        try:
            logging.info(f'getting file {object_name}...')
            response = self.client_instance.get_object(
                bucket_name=self.bucket_name, 
                object_name=object_name
            )
        except S3Error as e:
            logging.error(f'error getting file {object_name} with file type {file_type}. Error message: {e}')
        finally:
            if response != None:
                if file_type == 'json':
                    content = response.read()
                    logging.info('parsing json from the response....')
                    file = json.loads(content.decode('utf-8'))
                    logging.info(f'parsed json response .... {file}')
                elif file_type == 'image':
                    logging.info('parsing image from the response....')
                    file = Image.open(BytesIO(response.read()))
                response.close()
                response.release_conn()
            else:
                logging.error(f'file does not exists or theres an error getting the file {object_name}')
        
        return file
        
   