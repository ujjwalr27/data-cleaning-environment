FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all env files (they're at repo root on HF)
COPY . /app/data_cleaning_env

# Set Python path
ENV PYTHONPATH=/app

# HF Spaces requires non-root user
RUN useradd -m -u 1000 appuser
USER 1000

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "data_cleaning_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
