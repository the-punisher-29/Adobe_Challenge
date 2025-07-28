FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY input /app/input
RUN pip install --upgrade pip && \
    pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN mkdir -p /app/cache && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/cache')"
CMD ["python", "/app/src/round_1b.py"]
