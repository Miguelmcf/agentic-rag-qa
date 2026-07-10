FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Bundle the sample documents so the container is usable out of the box.
COPY data ./data

EXPOSE 8000

# Runs fully offline with the default echo LLM + hashing embeddings.
CMD ["uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
