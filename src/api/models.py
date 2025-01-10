import config
from dataclasses import dataclass
from pydantic import BaseModel
from services import MinioStorageService
from fastapi import UploadFile


class UploadFileModel(BaseModel):
    style_image: UploadFile
    content_image: UploadFile

@dataclass()
class ServiceRegistry:
    minio = MinioStorageService 