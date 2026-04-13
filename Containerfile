# Use a slim Python image
FROM python:3.11-slim

# Copy the uv binary from the official distroless image (2026 Best Practice)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment variables for uv and Python stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system dependencies (needed for certain AI packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use uv to install dependencies directly to the system Python
# This is significantly faster than pip
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Copy the rest of your app
COPY . .

CMD ["python", "app.py"]FROM python:3.11-slim

# 1. Optimize the OS environment
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. THE CRITICAL STEP: Isolated Copy
# We only copy requirements.txt. Podman checks the hash of THIS file only.
COPY requirements.txt .

# 3. Cacheable Installation
# As long as requirements.txt hasn't changed, Podman skips this 2-minute step.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code last
# This layer changes every time you save a file, but it's nearly instant.
COPY . .

CMD ["python", "app.py"]