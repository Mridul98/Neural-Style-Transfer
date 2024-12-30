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

    def __init__(self,host:str,access_key:str,secret_key:str):
       
        self.client_instance = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        logging.info(f'minio client instance id: {id(self.client_instance)}')

    def get_file(self, object_name:str, bucket_name:str):
        response = None
        try:
            response = self.client_instance.get_object(
                bucket_name=bucket_name, 
                object_name=object_name
            )
        finally:
            if response != None:
                response.close()
                response.release_conn()
     
    def upload_file(self,bucket_name,file):

        assert self.client_instance != None, "minio client instance should be instantiated first by calling get_instance() class method"
        self.client_instance.put_object(
            bucket_name=bucket_name,
            object_name=file.filename,
            data=file.file.read()
        )
