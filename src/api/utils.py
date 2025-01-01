import config
from models import ServiceRegistry, UploadFile

async def get_storage_service():
    """yield an object for interacting 
    with storage service

    Yields:
        StorageService: object that is 
        responsible for interacting with storage service
    """
    match config.STORAGE_TYPE:
        case "minio":
            yield ServiceRegistry.minio
        case _:
            raise NotImplementedError("storage type not implemented")

async def is_jpeg(file:UploadFile):
    """check whether uploaded file is an image
    type of 'jpeg'

    Args:
        file (UploadFile): uploaded file

    Returns:
        boolean: returns True if the file is an image type of jpeg else False
    """
    return True if file.content_type=='image/jpeg' else False