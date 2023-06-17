FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3.10-dev build-essential
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ['python3']
CMD ['services/server_221781.py']
