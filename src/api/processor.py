import uuid
from services import StorageService
from fastapi import UploadFile
from confluent_kafka.admin import AdminClient
from confluent_kafka import Producer
from models import UploadFileModel

class HTTPRequestProcessor:
    def __init__(self, style_image: UploadFile, content_image: UploadFile, storage_service: StorageService,  kafka_producer: Producer):
        
        self.request_payload = UploadFileModel(
            style_image=style_image,
            content_image=content_image,
            style_image_id=uuid.uuid4(),
            content_image_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            job_id=uuid.uuid4()
        )
    
    def _upload_files(self):
        try:
            self.request_payload.storage_service.upload_file(file=self.request_payload.style_image, prefix=self.request_payload.user_id)
            self.request_payload.storage_service.upload_file(file=self.request_payload.content_image, prefix=self.request_payload.user_id)
        except Exception as e:
            raise e
        
    def _enqueue_job(self):
        
        try:
            self.request_payload.kafka_producer.produce(
                topic='mridul',
                key=self.request_payload.job_id,
                value=self.request_payload.user_id
            )
        except Exception as e:
            raise e
    def process(self):
        self._upload_files()
        # Process the request and return a response