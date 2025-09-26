# Multi-stage Docker build for Fashion Item Classification API
# This Dockerfile creates a containerized deployment of our ML service

# Use official Python runtime as base image
FROM python:3.11-slim as base

# Set working directory inside the container
WORKDIR /app

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for PIL/Pillow image processing
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies first (for better caching)
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary project directories
# API source code
COPY api/ /app/api/

# Model architecture and configuration
COPY src/ /app/src/

# Pre-trained model weights
COPY models/ /app/models/

# Set Python path to include project directories
ENV PYTHONPATH=/app

# Expose the port that Flask runs on
EXPOSE 5001

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check to ensure the service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Command to run the Flask application with Gunicorn for production
# Using multiple workers to handle concurrent requests efficiently as described in slide 10 & 12
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "api.app:app"]

# Build instructions:
# docker build -t fashion-classifier-api .
# 
# Run instructions:
# docker run -p 5001:5001 fashion-classifier-api
