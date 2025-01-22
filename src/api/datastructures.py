from abc import ABC, abstractmethod
from uuid import UUID
from confluent_kafka import Producer
from pydantic import BaseModel, ConfigDict
from fastapi import UploadFile


class MetadataModel(BaseModel):
    style_image_id: UUID
    content_image_id: UUID
    user_id: UUID
    job_id: UUID

class StorageService(ABC):

    @abstractmethod
    def get_file():
        raise NotImplementedError()
    
    @abstractmethod
    def upload_file():
        raise NotImplementedError()
    
    @abstractmethod
    def upload_metadata(self,metadata:MetadataModel,prefix:str=None):
        raise NotImplementedError()

class UploadFileModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    style_image: UploadFile
    content_image: UploadFile
    style_image_id: UUID = None 
    content_image_id: UUID = None 
    user_id: UUID = None 
    job_id: UUID = None 
    storage_service: StorageService
    kafka_producer: Producer