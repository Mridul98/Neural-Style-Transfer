FROM python:3.10

WORKDIR /app

COPY ./ ./

RUN apt-get update
RUN python -m pip install --upgrade pip
RUN pip3 install -r 'broker_requirements.txt'

ENTRYPOINT [ "bash" ]