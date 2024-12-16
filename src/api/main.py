from inspect import trace
from operator import is_
import traceback
from fastapi import FastAPI, Depends, File, UploadFile, status, HTTPException, Response,Request
from services import StorageService, LocalStorageService
from typing import List

app = FastAPI()


async def get_storage_service():
    """yield an object for interacting 
    with storage service

    Yields:
        StorageService: object that is 
        responsible for interacting with storage service
    """
    yield LocalStorageService('./storage_dir')

async def is_jpeg(file:UploadFile):
    """check whether uploaded file is an image
    type of 'jpeg'

    Args:
        file (UploadFile): uploaded file

    Returns:
        boolean: returns True if the file is an image type of jpeg else False
    """
    return True if file.content_type=='image/jpeg' else False


@app.post('/nst', status_code= status.HTTP_201_CREATED,tags=['upload_images'])
async def submit_work(
    style_image: UploadFile= File(...),
    content_image: UploadFile = File(...), 
    storage_service: StorageService = Depends(get_storage_service)
    ):
    
    is_jpeg_style = await is_jpeg(style_image)
    is_content_style = await is_jpeg(content_image)
    
    if not all([is_jpeg_style,is_content_style]):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only Jpeg image is supported. Please upload file jpeg image'
        )
    else:

        try:

            storage_service.upload_file(style_image)
            storage_service.upload_file(content_image)

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND
            )

        return {
            'upload_status': f'file uploaded {style_image.filename} and {content_image.filename}'
        }

    
@app.get('/result/{job_id}',tags=['results'])
async def get_result(job_id:str):
    pass