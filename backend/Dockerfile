FROM python:3.9

WORKDIR /app

COPY ./ ./

RUN apt-get update

RUN pip3 install -r 'requirements.txt'

ENTRYPOINT ["python"]
