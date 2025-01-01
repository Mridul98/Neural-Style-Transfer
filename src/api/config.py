import os
from dotenv import load_dotenv

load_dotenv()

STORAGE_TYPE = os.getenv('STORAGE_TYPE')
MINIO_HOST = os.getenv('MINIO_HOST')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")