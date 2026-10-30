# One image for both the Streamlit console and the Celery page workers.
#   docker compose up --build        (see docker-compose.yml / README)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1 \
    HF_HOME=/opt/hf-cache \
    TESSERACT_CMD=/usr/bin/tesseract

# Tesseract OCR engine (system dependency of pytesseract) + runtime libs for OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# PyTorch (~200 MB) in its own cached layer, with retries: a dropped connection mid-download
# otherwise surfaces as a pip "hash mismatch" and fails the whole build.
RUN pip install --retries 10 --timeout 120 torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --retries 10 --timeout 120 -r requirements.txt

# Bake the two open-source models into the image: first run is fast and works offline.
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel as M; \
n='microsoft/trocr-small-handwritten'; TrOCRProcessor.from_pretrained(n); M.from_pretrained(n)" \
 && python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p /app/data/uploads

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
