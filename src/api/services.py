import os
import shutil
from abc import ABC, abstractmethod
from fastapi import UploadFile


class StorageService(ABC):

    @abstractmethod
    def upload_file():
        pass


class LocalStorageService(StorageService):

    def __init__(self, path: str):

        self.storage_path = path

        if not os.path.isdir(path):
            os.mkdir(path)

    def upload_file(self,file_to_upload:UploadFile):
        
        file_name = file_to_upload.filename
        final_file_path = os.path.join(self.storage_path,file_name)
        
        with open(final_file_path,'wb') as output_file:
            
            shutil.copyfileobj(file_to_upload.file,output_file)
            

        
