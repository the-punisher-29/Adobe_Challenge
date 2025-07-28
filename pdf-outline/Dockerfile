FROM python:3.10-slim

WORKDIR /app

# System dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY input /app/input

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "/app/src/round_1a.py"]
