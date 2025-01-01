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
    minio = MinioStorageService(
        host=config.MINIO_HOST,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        bucket_name=config.MINIO_BUCKET_NAME
    )