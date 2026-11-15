FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application assets
COPY . .

# Set environment variables for production
ENV PORT=7860
EXPOSE 7860

# Run Flask application via Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--workers", "2", "--timeout", "120"]
