FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment package
COPY . /app/data_cleaning_env

# Set Python path so import works
ENV PYTHONPATH=/app

# HF Spaces requires non-root user
RUN useradd -m -u 1000 appuser
USER 1000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# HF Spaces requires port 7860
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "data_cleaning_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

