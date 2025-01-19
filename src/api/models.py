from dataclasses import dataclass
from confluent_kafka import Producer
from services import StorageService
from pydantic import BaseModel
from fastapi import UploadFile


class UploadFileModel(BaseModel):
    style_image: UploadFile
    content_image: UploadFile
    style_image_id: str = None 
    content_image_id: str = None 
    user_id: str = None 
    job_id: str = None 
    storage_service: StorageService
    kafka_producer: Producer