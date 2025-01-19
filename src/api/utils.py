import config
from confluent_kafka import Producer
from services import MinioStorageService
from models import UploadFile

async def get_storage_service():
    """yield an object for interacting 
    with storage service

    Yields:
        StorageService: object that is 
        responsible for interacting with storage service
    """
    match config.STORAGE_TYPE:
        case "minio":
            yield MinioStorageService(
                host=config.MINIO_HOST,
                access_key=config.MINIO_ACCESS_KEY,
                secret_key=config.MINIO_SECRET_KEY,
                bucket_name=config.MINIO_BUCKET_NAME
            )
        case _:
            raise NotImplementedError("storage type not implemented")

async def get_kafka_producer():
    """yield an object for interacting with kafka producer

    Yields:
        Producer: object that is responsible for producing messages to kafka
    """
    yield Producer({'bootstrap.servers': '192.168.65.3:32486'})

async def is_jpeg(file:UploadFile):
    """check whether uploaded file is an image
    type of 'jpeg'

    Args:
        file (UploadFile): uploaded file

    Returns:
        boolean: returns True if the file is an image type of jpeg else False
    """
    return True if file.content_type=='image/jpeg' else False