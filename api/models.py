from pydantic import BaseModel
from fastapi import UploadFile


class UploadFileModel(BaseModel):

    style_image: UploadFile
    content_image: UploadFile