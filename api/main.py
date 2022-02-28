from inspect import trace
import traceback
from fastapi import FastAPI, Depends, File, UploadFile, status, HTTPException, Response
from services import StorageService, LocalStorageService


app = FastAPI()


async def get_storage_service():

    yield LocalStorageService('./storage_dir')


@app.post('/nst', status_code= status.HTTP_201_CREATED)
async def submit_work(
    style_image: UploadFile= File(...),
    content_image: UploadFile = File(...), 
    storage_service: StorageService = Depends(get_storage_service)
    ):
    
    try:
        storage_service.upload_file(style_image)
        storage_service.upload_file(content_image)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
        )
        return {'upload_status': f'Unable to upload files {style_image.filename} and {content_image.filename}'}

    return {'upload_status': f'file uploaded {style_image.filename} and {content_image.filename}'}

    

    

@app.get('/result/{job_id}')
async def get_result(job_id:str):
    pass