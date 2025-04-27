# Base image with Python 3.10
FROM python:3.10-slim AS base

# Prevents Python from writing .pyc and buffers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY . .

# Ensure log and data directories exist
RUN mkdir -p logs data/tmp

# Expose Streamlit default port
EXPOSE 8501

# Launch Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
