# Use Python 3.10 (works well with mediapipe / tensorflow)
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies used by OpenCV, ffmpeg, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer caching)
COPY requirements.txt .

# Upgrade pip and install deps
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port (match the port you use in uvicorn)
EXPOSE 7860

# Command to run your FastAPI app (adjust filename if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
