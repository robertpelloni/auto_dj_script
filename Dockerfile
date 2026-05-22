# Auto DJ Staging Environment Dockerfile (v7.3.0)
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    rubberband-cli \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose the Command Console port
EXPOSE 8000

# Default command to launch the GUI
CMD ["python", "auto_dj.py", "--gui"]
