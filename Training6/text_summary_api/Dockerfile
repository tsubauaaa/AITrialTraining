FROM python:3.8

WORKDIR /app
ADD . /app

RUN apt-get update -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["python", "main.py"]