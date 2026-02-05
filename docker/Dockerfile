# Dockerfile for Agent Seismograph
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY gatherer.py .
COPY config/ ./config/

# Create output directory
RUN mkdir -p /app/output

# Default command
CMD ["python", "gatherer.py", "--schedule"]
