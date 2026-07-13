FROM python:3.12-slim
RUN pip install --no-cache-dir pymilvus numpy requests
WORKDIR /app
COPY main.py .
CMD ["python", "main.py"]
