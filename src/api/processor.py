import uuid
import logging
from config import KAFKA_TOPIC
from fastapi import UploadFile
from confluent_kafka.admin import AdminClient
from confluent_kafka import Producer
from datastructures import UploadFileModel, MetadataModel, StorageService

class HTTPRequestProcessor:
    def __init__(self, style_image: UploadFile, content_image: UploadFile, storage_service: StorageService,  kafka_producer: Producer):
        
        self.request_payload = UploadFileModel(
            style_image=style_image,
            content_image=content_image,
            style_image_id=uuid.uuid4(),
            content_image_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            storage_service=storage_service,
            kafka_producer=kafka_producer)
        
        self.metadata = MetadataModel(
            style_image_id=self.request_payload.style_image_id,
            content_image_id=self.request_payload.content_image_id,
            user_id=self.request_payload.user_id,
            job_id=self.request_payload.job_id
        )
    
    def _upload_files(self):
        
        style_image_prefix = f'{self.request_payload.job_id}/{self.request_payload.style_image_id}'
        content_image_prefix = f'{self.request_payload.job_id}/{self.request_payload.content_image_id}'
        try:
            self.request_payload.storage_service.upload_file(
                file=self.request_payload.style_image, 
                prefix=style_image_prefix)
            self.request_payload.storage_service.upload_file(
                file=self.request_payload.content_image, 
                prefix=content_image_prefix)
            self.request_payload.storage_service.upload_metadata(
                metadata=self.metadata,
                prefix=self.request_payload.job_id)
        except Exception as e:
            raise e
        
    def _enqueue_job(self):
        
        try:
            self.request_payload.kafka_producer.produce(
                topic=KAFKA_TOPIC,
                key=str(self.request_payload.user_id),
                value=str(self.request_payload.job_id)
            )
            self.request_payload.kafka_producer.flush()
        except Exception as e:
            raise e
        
    def process(self):
        self._upload_files()
        self._enqueue_job()
        # Process the request and return a response