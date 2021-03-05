FROM python:3.8

WORKDIR /app
ADD . /app

RUN apt-get update -y && apt-get install -y libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]