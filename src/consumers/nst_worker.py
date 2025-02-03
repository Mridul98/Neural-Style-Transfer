import tomli
import logging
from config import MINIO_HOST, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME
from confluent_kafka import Consumer, KafkaException
from image_storage_client import ImageStorage
logging.basicConfig(level=logging.INFO,force=True)

class NSTWorker: 
    def __init__(self,config_path:str):
        logging.info("NSTWorker initialized")
        self.image_storage_client = ImageStorage(
            host=MINIO_HOST,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET_NAME
        )
        self.consumer_config = self._load_consumer_config(config_path)

    def _load_consumer_config(self,config_path:str):
        with open(config_path, "rb") as f:
            return tomli.load(f)['consumer']
       
    def _run_nst(self,content_image,style_image):
        pass

    def get_style_image(self,style_image):
        pass

    def get_content_image(self,content_image):
        pass

    def get_metadata_file(self,job_id,path='metadata.json'):
        path_prefix = f'/{job_id}/{path}'
        logging.info(f'getting metadata for job_id {job_id}...from {path_prefix}')
        return self.image_storage_client.get_file(file_type='json',object_name=path_prefix)
    
    def process(self):
        consumer = Consumer(self.consumer_config)
        consumer.subscribe(["nst"])
        logging.info("NSTWorker started")
        try:
            while True:

                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                elif msg.error():
                    logging.info("ERROR: %s".format(msg.error()))
                else:
                    # Extract the (optional) key and value, and logging.info.
                    logging.info("Consumed event from topic {topic}: key = {key:12} value = {value:12}".format(
                        topic=msg.topic(), key=msg.key().decode('utf-8'), value=msg.value().decode('utf-8')))
                    
                    logging.info(self.image_storage_client.get_file_stats(f"{msg.value().decode('utf-8')}/metadata.json"))
                    
                    metadata = self.get_metadata_file(msg.value().decode('utf-8'))
                    logging.info(f"metadata: {metadata.keys()}")
                    # logging.info(f"got list of objects {self.image_storage_client.list_files(msg.value().decode('utf-8'))}")
                    consumer.commit()
        except KeyboardInterrupt:
            pass
        finally:
            # Leave group and commit final offsets
            consumer.close()

if __name__ == "__main__":
    worker = NSTWorker("./kafka_configs.toml")
    worker.process()


    