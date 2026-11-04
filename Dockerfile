# Start from an official, small Python image
FROM python:3.12-slim

# Don't write .pyc files, print logs immediately, don't keep pip's download cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first, so this layer is cached when only code changes
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Then copy the application code
COPY . .

# Run as a regular user, not root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Let Docker check that the service is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]