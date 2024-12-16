FROM python:3.9

WORKDIR /app

COPY ./ ./

RUN apt-get update
RUN python -m pip install --upgrade pip
RUN pip3 install -r 'api_requirements.txt'
EXPOSE 8888
CMD ["uvicorn","main:app","--host", "0.0.0.0","--port","8888"]
