import os
import shutil
import logging
from abc import ABC, abstractmethod
from minio import Minio
from fastapi import UploadFile

class StorageService(ABC):

    @abstractmethod
    def get_file():
        raise NotImplementedError()
    
    @abstractmethod
    def upload_file():
        raise NotImplementedError()

class MinioStorageService(StorageService):

    def __init__(self,host:str,access_key:str,secret_key:str,bucket_name:str):
        
        self.client_instance = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        logging.info(f'minio client instance id: {id(self.client_instance)}')
        self._create_bucket(bucket_name=bucket_name)
    
    def _create_bucket(self,bucket_name):

        if self.client_instance.bucket_exists(bucket_name):
            logging.info(f"bucket {bucket_name} already exists!")
        else:
            self.client_instance.make_bucket(bucket_name)
            logging.info(f" created bucket named {bucket_name}")
        self.bucket_name = bucket_name
        
    def get_file(self, object_name:str):

        response = None
        try:
            logging.info(f'getting file {object_name}...')
            response = self.client_instance.get_object(
                bucket_name=self.bucket_name, 
                object_name=object_name
            )
        finally:
            if response != None:
                response.close()
                response.release_conn()
            else:
                logging.error(f'file does not exists or theres an error getting the file {object_name}')
        return response
     
    def upload_file(self,file: UploadFile):
        logging.info(f'uploading file: {file.filename}')
        try:
            self.client_instance.put_object(
                bucket_name=self.bucket_name,
                object_name=file.filename,
                data=file.file,
                length=-1,
                part_size=10*1024*1024
            )
        finally:
            logging.info(f'uploading file: {file.filename} succeeded')

