FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app

# increase timeout and retries
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=10

RUN pip install --upgrade pip \
 && pip install torch==2.3.1 \
 && pip install -r requirements.txt

CMD ["python3", "app.py"]
