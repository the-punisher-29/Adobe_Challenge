FROM python:3.10-slim

WORKDIR /app

# System dependencies for OCR and PDF tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY input /app/input

# Install torch manually
RUN pip install --upgrade pip && \
    pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install remaining dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Default command
CMD ["python", "/app/src/round1b.py"]
