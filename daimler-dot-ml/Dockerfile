# syntax=docker/dockerfile:1
# Образ для инференса. Перед сборкой выполните: python -m src.train
# (в каталоге проекта), чтобы появились artifacts/model.pth и feature_pipeline.json.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY src/ ./src/
COPY data/*.csv ./data/
COPY inference.ipynb .
COPY artifacts/ ./artifacts/

CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "inference.ipynb", \
     "--ExecutePreprocessor.timeout=1200", "--output", "inference_executed.ipynb"]
