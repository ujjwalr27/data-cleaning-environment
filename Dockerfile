FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Copy all env files into the package directory
COPY . /app/data_cleaning_env

# Install the package itself so imports work
COPY pyproject.toml /app/data_cleaning_env/pyproject.toml
RUN pip install --no-cache-dir -e /app/data_cleaning_env

# HF Spaces requires non-root user
RUN useradd -m -u 1000 appuser && \
  chown -R appuser:appuser /app
USER 1000

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "data_cleaning_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
